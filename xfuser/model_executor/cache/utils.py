"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import dataclasses
from typing import Dict, Optional, List
from xfuser.core.distributed import (
    get_sp_group,
    get_sequence_parallel_world_size,
)

import torch
import torch.nn.functional as F
from torch.nn import Module
from abc import ABC, abstractmethod
from typing import Tuple, List
import math


# --------- CacheContext --------- #
class CacheContext(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("default_coef", torch.tensor([1.0, 0.0]).cuda())
        self.register_buffer("flux_coef", torch.tensor([498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861]).cuda())
        
        self.register_buffer("original_hidden_states", None, persistent=False)
        self.register_buffer("original_encoder_hidden_states", None, persistent=False)
        self.register_buffer("hidden_states_residual", None, persistent=False)
        self.register_buffer("encoder_hidden_states_residual", None, persistent=False)
        self.register_buffer("modulated_inputs", None, persistent=False)
        
        # For FastCache
        self.register_buffer("prev_hidden_states", None, persistent=False)
        self.register_buffer("static_token_mask", None, persistent=False)
        
    def get_coef(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_coef")

#---------  CacheCallback  ---------#
@dataclasses.dataclass
class CacheState:
    transformer: Optional[torch.nn.Module] = None
    transformer_blocks: Optional[List[torch.nn.Module]] = None
    single_transformer_blocks: Optional[List[torch.nn.Module]] = None
    cache_context: Optional[CacheContext] = None
    rel_l1_thresh: float = 0.6
    return_hidden_states_first: bool = True
    use_cache: torch.Tensor = torch.tensor(False, dtype=torch.bool)
    num_steps: int = 8
    name: str = "default"


class CacheCallback:
    def on_init_end(self, state: CacheState, **kwargs): pass
    def on_forward_begin(self, state: CacheState, **kwargs): pass
    def on_forward_remaining_begin(self, state: CacheState, **kwargs): pass
    def on_forward_end(self, state: CacheState, **kwargs): pass


class CallbackHandler(CacheCallback):
    def __init__(self, callbacks: Optional[List[CacheCallback]] = None):
        self.callbacks = list(callbacks) if callbacks else []

    def trigger_event(self, event: str, state: CacheState):
        for cb in self.callbacks:
            getattr(cb, event)(state)

# --------- Vectorized Poly1D --------- #
class VectorizedPoly1D(Module):
    def __init__(self, coefficients: torch.Tensor):
        super().__init__()
        self.register_buffer("coefficients", coefficients)
        self.degree = len(coefficients) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x ** (self.degree - i))
        return result


class CachedTransformerBlocks(torch.nn.Module, ABC):
    def __init__(
        self,
        transformer_blocks: List[Module],
        single_transformer_blocks: Optional[List[Module]] = None,
        *,
        transformer: Optional[Module] = None,
        rel_l1_thresh: float = 0.6,
        return_hidden_states_first: bool = True,
        num_steps: int = -1,
        name: str = "default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(transformer_blocks)
        self.single_transformer_blocks = torch.nn.ModuleList(single_transformer_blocks) if single_transformer_blocks else None
        self.transformer = transformer
        self.register_buffer("cnt", torch.tensor(0).cuda())
        self.register_buffer("accumulated_rel_l1_distance", torch.tensor([0.0]).cuda())
        self.register_buffer("use_cache", torch.tensor(False, dtype=torch.bool).cuda())

        self.cache_context = CacheContext()
        self.callback_handler = CallbackHandler(callbacks)

        self.rel_l1_thresh = torch.tensor(rel_l1_thresh).cuda()
        self.return_hidden_states_first = return_hidden_states_first
        self.num_steps = num_steps
        self.name = name
        self.callback_handler.trigger_event("on_init_begin", self)

    @property
    def is_parallelized(self) -> bool:
        return get_sequence_parallel_world_size() > 1

    def all_reduce(self, input_: torch.Tensor, op=torch.distributed.ReduceOp.AVG) -> torch.Tensor:
        return get_sp_group().all_reduce(input_, op=op) if self.is_parallelized else input_

    def l1_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = (t1 - t2).abs().mean()
        norm = t1.abs().mean()
        diff, norm = self.all_reduce(diff.unsqueeze(0)), self.all_reduce(norm.unsqueeze(0))
        return (diff / norm).squeeze()

    @abstractmethod
    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor: pass

    @abstractmethod
    def get_start_idx(self) -> int: pass

    @abstractmethod
    def get_modulated_inputs(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, *args, **kwargs): pass

    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        for block in self.transformer_blocks[start_idx:]:
            hidden, encoder = block(hidden, encoder, *args, **kwargs)
            hidden, encoder = (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)

        if self.single_transformer_blocks:
            hidden = torch.cat([encoder, hidden], dim=1)
            for block in self.single_transformer_blocks:
                hidden = block(hidden, *args, **kwargs)
            encoder, hidden = hidden.split([encoder.shape[1], hidden.shape[1] - encoder.shape[1]], dim=1)

        self.cache_context.hidden_states_residual = hidden - self.cache_context.original_hidden_states
        self.cache_context.encoder_hidden_states_residual = encoder - self.cache_context.original_encoder_hidden_states
        return hidden, encoder

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        self.callback_handler.trigger_event("on_forward_begin", self)

        modulated, prev_modulated, orig_hidden, orig_encoder = \
            self.get_modulated_inputs(hidden_states, encoder_hidden_states, *args, **kwargs)

        self.cache_context.original_hidden_states = orig_hidden
        self.cache_context.original_encoder_hidden_states = orig_encoder

        self.use_cache = self.are_two_tensor_similar(prev_modulated, modulated, self.rel_l1_thresh) \
            if prev_modulated is not None else torch.tensor(False, dtype=torch.bool)

        self.callback_handler.trigger_event("on_forward_remaining_begin", self)
        if self.use_cache:
            hidden = hidden_states + self.cache_context.hidden_states_residual
            encoder = encoder_hidden_states + self.cache_context.encoder_hidden_states_residual
        else:
            hidden, encoder = self.process_blocks(self.get_start_idx(), orig_hidden, orig_encoder, *args, **kwargs)

        self.callback_handler.trigger_event("on_forward_end", self)
        return ((hidden, encoder) if self.return_hidden_states_first else (encoder, hidden))


class FBCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)

    def get_start_idx(self) -> int:
        return 1

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return self.l1_distance(t1, t2) < threshold

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_transformer_block(hidden_states, encoder_hidden_states, *args, **kwargs)
        hidden_states, encoder_hidden_states = (hidden_states, encoder_hidden_states) if self.return_hidden_states_first else (encoder_hidden_states, hidden_states)
        first_hidden_states_residual = hidden_states - original_hidden_states
        prev_first_hidden_states_residual = self.cache_context.modulated_inputs
        if not self.use_cache:
           self.cache_context.modulated_inputs = first_hidden_states_residual

        return first_hidden_states_residual, prev_first_hidden_states_residual, hidden_states, encoder_hidden_states


class TeaCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)
        self.rescale_func = VectorizedPoly1D(self.cache_context.get_coef(self.name))

    def get_start_idx(self) -> int:
        return 0

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        diff = self.l1_distance(t1, t2)
        new_accum = self.accumulated_rel_l1_distance + self.rescale_func(diff)
        reset_mask = (self.cnt == 0) or (self.cnt == self.num_steps - 1)
        self.use_cache = torch.logical_and(new_accum < threshold, torch.logical_not(reset_mask))
        self.accumulated_rel_l1_distance[0] = torch.where(self.use_cache, new_accum[0], 0.0)
        self.cnt = torch.where(self.cnt + 1 < self.num_steps, self.cnt + 1, 0)

        return self.use_cache

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        inp = hidden_states.clone()
        temb_ = kwargs.get("temb", None).clone()
        modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        prev_modulated = self.cache_context.modulated_inputs
        self.cache_context.modulated_inputs = modulated
        return modulated, prev_modulated, hidden_states, encoder_hidden_states


class FastCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.05,  # Default for FastCache is lower
        motion_threshold=0.1,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
        enable_enhanced_linear_approx=False,  # 新增：是否启用增强的线性近似算法
        significance_level=0.05,  # 新增：统计显著性水平
        enable_adacorrection: bool = False,  # 新增：是否启用AdaCorrection
        adacorr_gamma: float = 1.0,          # 新增：灵敏度 γ
        adacorr_lambda: float = 1.0,         # 新增：空间项权重 λ
        enable_token_merge: bool = False,    # 新增：是否启用Token Merging
        token_merge_k: int = 5,              # 新增：kNN的K值
        token_merge_lambda: float = 1.0,     # 新增：时间项权重λ（与adacorr_lambda不同）
        num_stages: int = 3,                 # 新增：多阶段金字塔编码的stage数量
        merge_ratio: float = 0.5,            # 新增：每个stage的token合并比例
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)
        
        # FastCache specific parameters
        self.motion_threshold = motion_threshold
        self.register_buffer("cache_hits", torch.tensor(0).cuda())
        self.register_buffer("total_steps", torch.tensor(0).cuda())
        
        # Initialize cache adaptation parameters
        self.beta0 = 0.01
        self.beta1 = 0.5
        self.beta2 = -0.002
        self.beta3 = 0.00005
        
        # 统计计算所需的参数
        self.confidence_level = 0.95  # 1 - alpha, 置信度
        self.z_score = 1.96  # z-score for 95% confidence

        # 统计Transformer嵌套层的数量和隐藏层大小
        self.num_layers = len(transformer_blocks)
        
        # Linear approximation for static tokens and transformer block outputs
        if hasattr(transformer_blocks[0], "config"):
            hidden_size = transformer_blocks[0].config.hidden_size
        else:
            # Estimate hidden size from the first block
            try:
                hidden_size = next(transformer_blocks[0].parameters()).shape[-1]
            except:
                hidden_size = 1024  # 默认值

        # 创建可学习的线性投影层（per-block linear approximation）
        self.block_projections = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size).cuda() 
            for _ in range(self.num_layers)
        ])
        
        # 创建空间token减少模块的线性投影
        self.spatial_projection = torch.nn.Linear(hidden_size, hidden_size).cuda()
        
        # 新增：增强线性近似算法的参数
        self.enable_enhanced_linear_approx = enable_enhanced_linear_approx
        self.significance_level = significance_level
        
        if self.enable_enhanced_linear_approx:
            # 为增强算法创建额外的线性投影层
            self.enhanced_block_projections = torch.nn.ModuleList([
                torch.nn.Linear(hidden_size, hidden_size).cuda() 
                for _ in range(self.num_layers)
            ])
            
            # 静态token的专用线性投影
            self.static_token_projection = torch.nn.Linear(hidden_size, hidden_size).cuda()
            
            # 统计阈值计算参数
            self.chi2_thresholds = {}  # 缓存卡方分布阈值
            
            print(f"Enhanced Linear Approximation Algorithm enabled with significance_level={significance_level}")
        
        # 新增：AdaCorrection 参数
        self.enable_adacorrection = enable_adacorrection
        self.adacorr_gamma = adacorr_gamma
        self.adacorr_lambda = adacorr_lambda
        
        # AdaCorrection cache storage for per-layer cached outputs
        if self.enable_adacorrection:
            self.register_buffer("adacorr_cache", None, persistent=False)
            self.register_buffer("adacorr_cache_timestep", None, persistent=False)
            print(f"AdaCorrection enabled with gamma={adacorr_gamma}, lambda={adacorr_lambda}")
        
        # 新增：Token Merging 参数
        self.enable_token_merge = enable_token_merge
        self.token_merge_k = token_merge_k
        self.token_merge_lambda = token_merge_lambda
        self.num_stages = num_stages
        self.merge_ratio = merge_ratio
        
        # Token merging storage for multi-stage processing
        if self.enable_token_merge:
            self.register_buffer("stage_outputs", None, persistent=False)  # Z[s] for each stage
            self.register_buffer("merge_masks", None, persistent=False)   # M[s] for each stage
            print(f"Token Merging enabled with K={token_merge_k}, lambda={token_merge_lambda}, stages={num_stages}")

    def get_chi2_threshold(self, n, d, alpha):
        """计算卡方分布阈值 χ²_{ND, 1-α}"""
        dof = n * d  # degrees of freedom
        key = (n, d, alpha)
        
        if key not in self.chi2_thresholds:
            # 对于大自由度，使用正态分布近似卡方分布
            # χ²_{ND, 1-α} ≈ ND + z_{1-α} * sqrt(2*ND)
            z_alpha = torch.erfinv(2 * (1 - alpha) - 1) * math.sqrt(2)
            chi2_threshold = dof + z_alpha * math.sqrt(2 * dof)
            self.chi2_thresholds[key] = chi2_threshold
        
        return self.chi2_thresholds[key]

    def enhanced_block_level_linear_approximation(self, hidden_states, prev_hidden_states, layer_idx):
        """
        增强的块级线性近似算法
        
        Algorithm: FastCache Linear Approximation Framework
        Input: Hidden state H_t, previous H_{t-1}, learnable parameters {W, b}
        Output: Approximated hidden state H_t^L
        """
        if prev_hidden_states is None:
            return hidden_states, False
        
        # 计算相对变化 δ_{t,l} = ||H_{t,l-1} - H_{t-1,l-1}||_F / ||H_{t-1,l-1}||_F
        diff_norm = torch.norm(hidden_states - prev_hidden_states, p='fro')
        prev_norm = torch.norm(prev_hidden_states, p='fro')
        
        if prev_norm == 0:
            return hidden_states, False
        
        delta = diff_norm / prev_norm
        
        # 计算统计阈值
        n, d = hidden_states.shape[0], hidden_states.shape[1]
        chi2_threshold = self.get_chi2_threshold(n, d, self.significance_level)
        statistical_threshold = math.sqrt(chi2_threshold / (n * d))
        
        # 判断是否使用线性近似：δ_{t,l}² ≤ χ²_{ND,1-α}/ND
        use_linear_approx = (delta ** 2) <= (chi2_threshold / (n * d))
        
        if use_linear_approx:
            # 应用线性近似：H_{t,l} = W_block_l × H_{t,l-1} + b_block_l
            approximated_hidden = self.enhanced_block_projections[layer_idx](hidden_states)
            return approximated_hidden, True
        else:
            # 完整transformer计算
            return hidden_states, False

    def enhanced_token_level_linear_approximation(self, hidden_states, prev_hidden_states, encoder_hidden_states=None):
        """
        增强的token级线性近似算法
        
        Algorithm: FastCache Linear Approximation with Masking
        Input: Token embedding X_t, previous X_{t-1}, Previous hidden states {H_{t-1,l}}
        Output: Final hidden state H_{t,L}
        """
        if prev_hidden_states is None:
            return hidden_states, encoder_hidden_states, None, None, None, None
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # === Token-Level Saliency Mask ===
        # 计算显著性：S_t = ||X_t - X_{t-1}||²
        token_diffs = (hidden_states - prev_hidden_states) ** 2
        saliency = token_diffs.sum(dim=-1)  # [batch_size, seq_len]
        
        # Token mask: M_token = {i | S_t[i] > τ_motion}, S_token = {i | S_t[i] ≤ τ_motion}
        motion_mask = saliency > self.motion_threshold  # [batch_size, seq_len]
        static_mask = ~motion_mask
        
        # 如果没有静态token，直接返回
        if not static_mask.any():
            return hidden_states, encoder_hidden_states, None, None, motion_mask, static_mask
        
        # 预计算静态token的输出：H_static = W_static × X_t[S_token] + b_static
        static_hidden = hidden_states.clone()
        static_hidden[static_mask] = self.static_token_projection(hidden_states[static_mask])
        
        # 运动token通过完整transformer处理
        motion_hidden = hidden_states.clone()
        
        # 处理encoder_hidden_states（如果存在）
        if encoder_hidden_states is not None:
            static_encoder = encoder_hidden_states.clone()
            static_encoder[static_mask] = self.static_token_projection(encoder_hidden_states[static_mask])
            motion_encoder = encoder_hidden_states.clone()
        else:
            static_encoder = None
            motion_encoder = None
        
        return motion_hidden, motion_encoder, static_hidden, static_encoder, motion_mask, static_mask

    def compute_adacorr_offset_score(self, current_hidden: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute AdaCorrection offset score S_t^l according to the paper:
        
        S_t^l = ||Δ_temp(t)||^2 + λ * ||∇_x h_t^l||^2
        
        Where:
        - Δ_temp(t) = (1/BP) * Σ_{b,i} ||h_t^l[b,i,:] - h_{t-1}^l[b,i,:]||_2  (temporal deviation)
        - ||∇_x h_t^l||^2 ≈ Δ_spatial(t) = (1/BP) * Σ_{b,i} sqrt(Var_d(h_t^l[b,i,d]))  (spatial variation)
        
        Args:
            current_hidden: h_t^l ∈ R^{B×P×D} - current hidden state at layer l, timestep t
            prev_hidden: h_{t-1}^l ∈ R^{B×P×D} - previous hidden state at layer l, timestep t-1
            
        Returns:
            S_t^l: offset score (scalar tensor)
        """
        if prev_hidden is None:
            return torch.tensor(0.0, device=current_hidden.device)
        
        B, P, D = current_hidden.shape
        
        # Compute temporal deviation: Δ_temp(t) = (1/BP) * Σ ||h_t[b,i,:] - h_{t-1}[b,i,:]||_2
        # For each token [b,i], compute L2 norm across channel dimension
        token_diff = current_hidden - prev_hidden  # [B, P, D]
        token_diff_norm = torch.norm(token_diff, p=2, dim=-1)  # [B, P] - L2 norm per token
        delta_temp = token_diff_norm.mean()  # scalar: average across all tokens
        
        # Compute spatial variation: Δ_spatial(t) = (1/BP) * Σ sqrt(Var_d(h_t[b,i,d]))
        # For each token [b,i], compute variance across channel dimension, then take sqrt
        spatial_var = current_hidden.var(dim=-1, unbiased=False)  # [B, P] - variance per token
        delta_spatial = torch.sqrt(spatial_var.clamp_min(1e-12)).mean()  # scalar: sqrt of variance, then average
        
        # Compute offset score: S_t^l = ||Δ_temp(t)||^2 + λ * ||∇_x h_t^l||^2
        # According to paper, ||∇_x h_t^l||^2 is approximated by Δ_spatial(t)
        score = (delta_temp ** 2) + self.adacorr_lambda * delta_spatial
        
        return score

    def blend_with_adacorrection(self, cached_hidden: torch.Tensor, fresh_hidden: torch.Tensor, offset_score: torch.Tensor) -> torch.Tensor:
        """根据 λ_t^l = clip(γ * S_t^l, 0, 1) 对 cached 与 fresh 做插值"""
        lam = torch.clamp(self.adacorr_gamma * offset_score, 0.0, 1.0)
        # 广播到 hidden 形状
        while lam.dim() < fresh_hidden.dim():
            lam = lam.unsqueeze(0)
        return (1 - lam) * cached_hidden + lam * fresh_hidden

    def compute_spatial_density(self, hidden_states: torch.Tensor, k: int = None) -> torch.Tensor:
        """
        Compute spatial density ρsp,i for each token using kNN.
        
        ρsp,i = exp(-1/K * Σ_{j∈kNN(i)} ||ht,i - ht,j||²)
        
        Args:
            hidden_states: [B, P, D] hidden states
            k: number of nearest neighbors (default: self.token_merge_k)
            
        Returns:
            ρsp: [B, P] spatial density scores
        """
        if k is None:
            k = self.token_merge_k
        
        B, P, D = hidden_states.shape
        
        # Reshape to [B*P, D] for efficient pairwise distance computation
        h_flat = hidden_states.view(B * P, D)  # [B*P, D]
        
        # Compute pairwise squared distances: ||ht,i - ht,j||²
        # Using efficient matrix multiplication: ||a-b||² = ||a||² + ||b||² - 2*a*b
        h_norm_sq = (h_flat ** 2).sum(dim=-1, keepdim=True)  # [B*P, 1]
        pairwise_dist_sq = h_norm_sq + h_norm_sq.T - 2 * torch.mm(h_flat, h_flat.T)  # [B*P, B*P]
        
        # For each token, find k nearest neighbors (excluding self)
        # Add large value to diagonal to exclude self
        pairwise_dist_sq.fill_diagonal_(float('inf'))
        
        # Find k nearest neighbors for each token
        k_nearest_dist_sq, _ = torch.topk(pairwise_dist_sq, k=min(k, P-1), dim=-1, largest=False)  # [B*P, k]
        
        # Compute spatial density: ρsp,i = exp(-1/K * Σ ||ht,i - ht,j||²)
        sum_dist_sq = k_nearest_dist_sq.sum(dim=-1)  # [B*P]
        rho_sp = torch.exp(-sum_dist_sq / k)  # [B*P]
        
        # Reshape back to [B, P]
        rho_sp = rho_sp.view(B, P)
        
        return rho_sp

    def compute_temporal_saliency_tokenwise(self, hidden_states: torch.Tensor, prev_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute token-wise temporal saliency ρtm,i.
        
        ρtm,i = ||ht,i - ht-1,i||₂
        
        Args:
            hidden_states: [B, P, D] current hidden states
            prev_hidden_states: [B, P, D] previous hidden states
            
        Returns:
            ρtm: [B, P] temporal saliency scores
        """
        if prev_hidden_states is None:
            return torch.zeros(hidden_states.shape[:2], device=hidden_states.device)
        
        # Compute L2 norm per token: ||ht,i - ht-1,i||₂
        token_diff = hidden_states - prev_hidden_states  # [B, P, D]
        rho_tm = torch.norm(token_diff, p=2, dim=-1)  # [B, P]
        
        return rho_tm

    def compute_token_importance_score(self, hidden_states: torch.Tensor, prev_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute combined token importance score Si.
        
        Si = ρsp,i · (1 + λ · ρtm,i)
        
        Args:
            hidden_states: [B, P, D] current hidden states
            prev_hidden_states: [B, P, D] previous hidden states
            
        Returns:
            S: [B, P] importance scores
        """
        # Compute spatial density
        rho_sp = self.compute_spatial_density(hidden_states)  # [B, P]
        
        # Compute temporal saliency
        rho_tm = self.compute_temporal_saliency_tokenwise(hidden_states, prev_hidden_states)  # [B, P]
        
        # Normalize temporal saliency to [0, 1] range for stability
        if rho_tm.max() > 0:
            rho_tm = rho_tm / (rho_tm.max() + 1e-8)
        
        # Combined importance score: Si = ρsp,i · (1 + λ · ρtm,i)
        S = rho_sp * (1 + self.token_merge_lambda * rho_tm)
        
        return S

    def local_ctm(self, hidden_states: torch.Tensor, importance_scores: torch.Tensor, target_ratio: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Local Clustering-based Token Merge (LocalCTM).
        
        Groups tokens into clusters and merges tokens within each cluster via weighted averaging:
        h̃_k = Σ_{j∈Ck} (Sj * ht,j) / Σ_{j∈Ck} Sj
        
        Args:
            hidden_states: [B, P, D] hidden states to merge
            importance_scores: [B, P] importance scores S
            target_ratio: ratio of tokens to keep (default: self.merge_ratio)
            
        Returns:
            merged_hidden: [B, P', D] merged hidden states (P' ≈ P * target_ratio)
            merge_mask: [B, P] boolean mask indicating which original tokens were kept
        """
        if target_ratio is None:
            target_ratio = self.merge_ratio
        
        B, P, D = hidden_states.shape
        
        # Determine number of tokens to keep
        P_target = max(1, int(P * target_ratio))
        
        # For simplicity, use importance-based selection: keep top P_target tokens
        # In a full implementation, this would use clustering (e.g., k-means or hierarchical)
        _, top_indices = torch.topk(importance_scores, k=P_target, dim=-1)  # [B, P_target]
        
        # Create merge mask: True for tokens to keep
        merge_mask = torch.zeros(B, P, dtype=torch.bool, device=hidden_states.device)
        for b in range(B):
            merge_mask[b, top_indices[b]] = True
        
        # For kept tokens, use weighted average of their cluster
        # Simplified: for each kept token, average with its nearest neighbors
        merged_hidden = hidden_states.clone()
        
        # Compute pairwise distances for clustering
        h_flat = hidden_states.view(B * P, D)
        h_norm_sq = (h_flat ** 2).sum(dim=-1, keepdim=True)
        pairwise_dist_sq = h_norm_sq + h_norm_sq.T - 2 * torch.mm(h_flat, h_flat.T)
        
        # For each batch
        for b in range(B):
            batch_start = b * P
            batch_end = (b + 1) * P
            
            # Get kept token indices for this batch
            kept_indices = top_indices[b]  # [P_target]
            
            # For each kept token, merge with its nearest neighbors
            for kept_idx in kept_indices:
                # Find nearest neighbors (including self)
                global_kept_idx = batch_start + kept_idx.item()
                dists = pairwise_dist_sq[global_kept_idx, batch_start:batch_end]  # [P]
                
                # Find k nearest neighbors
                k = min(self.token_merge_k, P)
                _, nn_indices = torch.topk(dists, k=k, largest=False)
                
                # Weighted average using importance scores
                cluster_indices = nn_indices
                cluster_weights = importance_scores[b, cluster_indices]  # [k]
                cluster_weights = cluster_weights / (cluster_weights.sum() + 1e-8)
                
                # Weighted average: h̃_k = Σ (Sj * ht,j) / Σ Sj
                merged_token = (hidden_states[b, cluster_indices] * cluster_weights.unsqueeze(-1)).sum(dim=0)  # [D]
                merged_hidden[b, kept_idx] = merged_token
        
        # Select only kept tokens
        merged_hidden = merged_hidden[merge_mask.view(B, P)].view(B, P_target, D)
        
        return merged_hidden, merge_mask

    def unpool_tokens(self, hidden_states: torch.Tensor, merge_mask: torch.Tensor, original_size: int) -> torch.Tensor:
        """
        Unpool tokens back to original size using merge mask.
        
        This is a simplified unpooling: each merged token is broadcast to its original cluster.
        
        Args:
            hidden_states: [B, P', D] merged hidden states
            merge_mask: [B, P] boolean mask indicating which original tokens were kept
            original_size: original number of tokens P
            
        Returns:
            unpooled: [B, P, D] unpooled hidden states
        """
        B, P_merged, D = hidden_states.shape
        P_original = original_size
        
        # Create unpooled tensor
        unpooled = torch.zeros(B, P_original, D, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # For each batch, place merged tokens at their original positions
        for b in range(B):
            kept_indices = torch.where(merge_mask[b])[0]  # [P_merged]
            if len(kept_indices) > 0:
                # Map kept indices to their positions in merged hidden states
                # Since merge_mask indicates which original tokens were kept,
                # we need to find the corresponding positions in hidden_states
                kept_positions = torch.arange(len(kept_indices), device=hidden_states.device)
                unpooled[b, kept_indices] = hidden_states[b, kept_positions]
            
            # For non-kept tokens, use nearest neighbor interpolation
            if len(kept_indices) < P_original:
                # Find nearest kept token for each non-kept token
                non_kept_indices = torch.where(~merge_mask[b])[0]
                if len(non_kept_indices) > 0 and len(kept_indices) > 0:
                    # Simple interpolation: use the closest kept token
                    for non_kept_idx in non_kept_indices:
                        # Find closest kept token
                        dists = torch.abs(kept_indices.float() - non_kept_idx.float())
                        closest_kept_idx = kept_indices[dists.argmin()]
                        # Find position of closest_kept_idx in kept_indices
                        closest_pos = (kept_indices == closest_kept_idx).nonzero(as_tuple=True)[0]
                        if len(closest_pos) > 0:
                            unpooled[b, non_kept_idx] = hidden_states[b, closest_pos[0]]
        
        return unpooled

    def multi_stage_pyramidal_encoding(self, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Multi-Stage Pyramidal Backbone Encoding with CTM Downsampling.
        
        Algorithm 2 Part 1: Pyramidal Backbone Encoding
        For each stage s = 1 to S:
            - Process through Ls transformer blocks
            - Apply CTM downsampling (except last stage)
            - Store stage output Z[s] and merge mask M[s]
        
        Args:
            hidden: [B, P, D] input hidden states
            encoder: [B, P, D] encoder hidden states
            *args, **kwargs: additional arguments for transformer blocks
            
        Returns:
            final_hidden: [B, P_final, D] final hidden states after all stages
            stage_outputs: List of [B, P_s, D] outputs for each stage Z[s]
            merge_masks: List of [B, P_s] merge masks M[s] for each stage
        """
        B, P_initial, D = hidden.shape
        prev_hidden = self.cache_context.prev_hidden_states
        
        stage_outputs = []
        merge_masks = []
        
        current_hidden = hidden
        current_encoder = encoder
        current_size = P_initial
        
        # Compute initial token importance score
        importance_scores = self.compute_token_importance_score(current_hidden, prev_hidden)
        
        # Process through S stages
        for s in range(1, self.num_stages + 1):
            # Determine number of blocks for this stage (distribute blocks across stages)
            blocks_per_stage = self.num_layers // self.num_stages
            start_block = (s - 1) * blocks_per_stage
            end_block = s * blocks_per_stage if s < self.num_stages else self.num_layers
            
            # Process through blocks in this stage
            for l in range(start_block, end_block):
                if l < len(self.transformer_blocks):
                    block = self.transformer_blocks[l]
                    
                    # Compute relative change
                    if prev_hidden is not None:
                        delta = self.compute_relative_change(current_hidden, prev_hidden)
                        n, d = current_hidden.shape[0], current_hidden.shape[1]
                        chi2_threshold = self.get_chi2_threshold(n, d, self.significance_level)
                        statistical_threshold = math.sqrt(chi2_threshold / (n * d))
                        
                        # Decide whether to use linear approximation
                        if (delta ** 2) <= (chi2_threshold / (n * d)):
                            # Linear approximation: H_{t,l} = W_{s,l} H_{t,l-1} + b_{s,l}
                            current_hidden = self.block_projections[l](current_hidden)
                        else:
                            # Full computation: H_{t,l} = Blocks,l(H_{t,l-1})
                            current_hidden, current_encoder = block(current_hidden, current_encoder, *args, **kwargs)
                            current_hidden, current_encoder = (current_hidden, current_encoder) if self.return_hidden_states_first else (current_encoder, current_hidden)
                    else:
                        # First timestep: full computation
                        current_hidden, current_encoder = block(current_hidden, current_encoder, *args, **kwargs)
                        current_hidden, current_encoder = (current_hidden, current_encoder) if self.return_hidden_states_first else (current_encoder, current_hidden)
                    
                    # Update prev_hidden for next layer
                    prev_hidden = current_hidden.detach().clone()
            
            # Store stage output Z[s] = H_{t,Ls}
            stage_outputs.append(current_hidden.detach().clone())
            
            # CTM Downsampling (except last stage)
            if s < self.num_stages:
                # Recompute importance scores for current hidden states
                importance_scores = self.compute_token_importance_score(current_hidden, prev_hidden)
                
                # Apply LocalCTM downsampling
                current_hidden, merge_mask = self.local_ctm(current_hidden, importance_scores, target_ratio=self.merge_ratio)
                if current_encoder is not None:
                    # Apply same merge mask to encoder
                    B_curr, P_curr, D_curr = current_encoder.shape
                    merge_mask_encoder = merge_mask[:, :P_curr] if merge_mask.shape[1] >= P_curr else merge_mask
                    current_encoder, _ = self.local_ctm(current_encoder, importance_scores[:, :P_curr] if importance_scores.shape[1] >= P_curr else importance_scores, target_ratio=self.merge_ratio)
                
                # Store merge mask M[s]
                merge_masks.append(merge_mask)
                current_size = current_hidden.shape[1]
        
        return current_hidden, stage_outputs, merge_masks

    def multi_stage_token_aggregation(self, final_hidden: torch.Tensor, stage_outputs: List[torch.Tensor], merge_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Multi-Stage Token Aggregation (MTA) for upsampling.
        
        Algorithm 2 Part 2: Multi-stage Token Aggregation
        H_agg ← H_{t,LS}
        For s = S-1 down to 1:
            H_agg ← Unpool(H_agg, M[s])
            H_agg ← H_agg + Z[s]
        
        Args:
            final_hidden: [B, P_final, D] final hidden states from last stage
            stage_outputs: List of [B, P_s, D] outputs Z[s] for each stage
            merge_masks: List of [B, P_s] merge masks M[s] for each stage
            
        Returns:
            aggregated: [B, P_initial, D] aggregated hidden states
        """
        # Start with final stage output
        H_agg = final_hidden
        
        # Aggregate from last stage to first (reverse order)
        for s in range(len(stage_outputs) - 2, -1, -1):  # S-1 down to 1
            # Unpool to match stage s size
            stage_size = stage_outputs[s].shape[1]
            H_agg = self.unpool_tokens(H_agg, merge_masks[s], stage_size)
            
            # Add stage output: H_agg ← H_agg + Z[s]
            # Ensure shapes match
            if H_agg.shape[1] == stage_outputs[s].shape[1]:
                H_agg = H_agg + stage_outputs[s]
            else:
                # If sizes don't match, interpolate
                if H_agg.shape[1] < stage_outputs[s].shape[1]:
                    # Upsample H_agg to match stage_outputs[s]
                    H_agg = F.interpolate(H_agg.transpose(1, 2), size=stage_outputs[s].shape[1], mode='linear', align_corners=False).transpose(1, 2)
                else:
                    # Downsample stage_outputs[s] to match H_agg
                    stage_interp = F.interpolate(stage_outputs[s].transpose(1, 2), size=H_agg.shape[1], mode='linear', align_corners=False).transpose(1, 2)
                    H_agg = H_agg + stage_interp
        
        return H_agg

    def enhanced_process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        """
        增强的块处理算法，结合块级和token级线性近似，且可选执行 AdaCorrection 纠偏插值
        如果启用Token Merging，则使用多阶段金字塔编码
        """
        # 如果启用Token Merging，使用多阶段金字塔编码
        if self.enable_token_merge:
            final_hidden, stage_outputs, merge_masks = self.multi_stage_pyramidal_encoding(hidden, encoder, *args, **kwargs)
            aggregated_hidden = self.multi_stage_token_aggregation(final_hidden, stage_outputs, merge_masks)
            
            # 处理single_transformer_blocks（如果存在）
            if self.single_transformer_blocks and encoder is not None:
                aggregated_hidden = torch.cat([encoder, aggregated_hidden], dim=1)
                for block in self.single_transformer_blocks:
                    aggregated_hidden = block(aggregated_hidden, *args, **kwargs)
                encoder, aggregated_hidden = aggregated_hidden.split([encoder.shape[1], aggregated_hidden.shape[1] - encoder.shape[1]], dim=1)
            
            return aggregated_hidden, encoder
        
        if not self.enable_enhanced_linear_approx and not self.enable_adacorrection:
            return self.process_blocks(start_idx, hidden, encoder, *args, **kwargs)
        
        # 获取previous hidden states
        prev_hidden_states = self.cache_context.prev_hidden_states
        
        # 若启用空间/token级线性近似，先执行
        if self.enable_enhanced_linear_approx:
            result = self.enhanced_token_level_linear_approximation(hidden, prev_hidden_states, encoder)
            motion_hidden, motion_encoder, static_hidden, static_encoder, motion_mask, static_mask = result
            if motion_mask is None or static_mask is None:
                # 回退为原始流程（仍可使用 AdaCorrection）
                motion_hidden, motion_encoder = hidden, encoder
                static_hidden = static_encoder = None
        else:
            motion_hidden, motion_encoder = hidden, encoder
            static_hidden = static_encoder = None
            motion_mask = static_mask = None
        
        current_hidden = motion_hidden
        current_encoder = motion_encoder
        
        for i, block in enumerate(self.transformer_blocks[start_idx:], start=start_idx):
            # 计算 cached 路径（线性近似）和 fresh 路径（完整块）
            cached_hidden_i = self.block_projections[i](current_hidden)
            fresh_hidden_i, fresh_encoder_i = block(current_hidden, current_encoder, *args, **kwargs)
            fresh_hidden_i, fresh_encoder_i = (fresh_hidden_i, fresh_encoder_i) if self.return_hidden_states_first else (fresh_encoder_i, fresh_hidden_i)
            
            if self.enable_adacorrection and prev_hidden_states is not None:
                # 计算偏移分数并进行插值
                offset_score = self.compute_adacorr_offset_score(current_hidden, prev_hidden_states)
                blended_hidden = self.blend_with_adacorrection(cached_hidden_i, fresh_hidden_i, offset_score)
                current_hidden, current_encoder = blended_hidden, fresh_encoder_i
            else:
                # 若未启用 AdaCorrection，则依据增强线性近似的判定决定
                if self.enable_enhanced_linear_approx:
                    approximated_hidden, used_approx = self.enhanced_block_level_linear_approximation(current_hidden, prev_hidden_states, i)
                    if used_approx:
                        current_hidden = approximated_hidden
                        self.cache_hits += 1
                        continue
                # 默认采用 fresh 路径
                current_hidden, current_encoder = fresh_hidden_i, fresh_encoder_i
        
        # 处理 single_transformer_blocks（如果存在）
        if self.single_transformer_blocks and current_encoder is not None:
            current_hidden = torch.cat([current_encoder, current_hidden], dim=1)
            for block in self.single_transformer_blocks:
                current_hidden = block(current_hidden, *args, **kwargs)
            current_encoder, current_hidden = current_hidden.split([current_encoder.shape[1], current_hidden.shape[1] - current_encoder.shape[1]], dim=1)
        
        # 最终组合运动/静态 token（若有）
        if motion_mask is not None and static_mask is not None and static_hidden is not None:
            final_hidden = hidden.clone()
            final_hidden[motion_mask] = current_hidden[motion_mask]
            final_hidden[static_mask] = static_hidden[static_mask]
            final_encoder = encoder.clone() if encoder is not None else None
            if final_encoder is not None and current_encoder is not None and static_encoder is not None:
                final_encoder[motion_mask] = current_encoder[motion_mask]
                final_encoder[static_mask] = static_encoder[static_mask]
            return final_hidden, final_encoder
        
        return current_hidden, current_encoder

    def get_start_idx(self) -> int:
        return 0  # Process all blocks when not caching
    
    def get_adaptive_threshold(self, variance_score, timestep=None):
        """Calculate adaptive threshold based on variance and current timestep"""
        if timestep is None:
            timestep = self.cnt
            
        normalized_timestep = timestep / 1000.0  # Normalize timestep to [0,1] range
        return (self.beta0 + 
                self.beta1 * variance_score + 
                self.beta2 * normalized_timestep + 
                self.beta3 * normalized_timestep**2)

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        """Using FastCache's relative change metric for caching decision"""
        if t1 is None or t2 is None:
            return torch.tensor(False, dtype=torch.bool).cuda()
            
        # 计算相对变化（Frobenius范数）
        # δ_{t,l} = ||H_{t,l-1} - H_{t-1,l-1}||_F / ||H_{t-1,l-1}||_F
        diff_norm = torch.norm(t1 - t2, p='fro')
        prev_norm = torch.norm(t2, p='fro')
        
        # 避免除以零
        if prev_norm == 0:
            delta = torch.tensor(float('inf')).cuda()
        else:
            delta = (diff_norm / prev_norm)
            
        # Update total steps counter
        self.total_steps += 1
        
        # 计算统计阈值
        # (ND) · δ_{t,l}^2 ~ χ^2_{ND}
        # 对于大自由度，可以用正态分布近似卡方分布
        n, d = t1.shape[0], t1.shape[1]  # token count, hidden dim
        dof = n * d  # degrees of freedom
        
        # chi2_threshold = χ^2_{ND, 1-α}
        chi2_threshold = dof + self.z_score * math.sqrt(2 * dof)
        
        # 根据公式计算阈值: δ_{t,l}^2 ≤ χ^2_{ND, 1-α}/ND
        statistical_threshold = math.sqrt(chi2_threshold / dof)
        
        # Adaptive threshold based on variance and timestep
        adaptive_threshold = self.get_adaptive_threshold(delta, self.cnt)
        
        # Final threshold combines statistical validity with adaptive behavior
        final_threshold = max(threshold, min(statistical_threshold, adaptive_threshold))
        
        # Cache decision - 如果相对变化小于阈值，则使用缓存
        use_cache = delta <= final_threshold
        
        # Update cache hits counter
        self.cache_hits += use_cache.int()
        
        return use_cache

    def compute_motion_saliency(self, hidden_states):
        """Compute motion saliency for spatial token reduction
           S_t = ||X_t - X_{t-1}||_2^2 
        """
        if self.cache_context.prev_hidden_states is None:
            return torch.ones(hidden_states.shape[1], device=hidden_states.device)
            
        # 计算空间token的显著性（逐token计算差异平方和）
        token_diffs = (hidden_states - self.cache_context.prev_hidden_states)**2
        
        # 沿特征维度求和得到每个token的显著性
        token_saliency = token_diffs.sum(dim=-1).squeeze(0)
        
        # 归一化显著性
        if token_saliency.max() > 0:
            token_saliency = token_saliency / token_saliency.max()
            
        return token_saliency

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        # Store current hidden states for later comparisons
        prev_hidden_states = self.cache_context.prev_hidden_states
        
        # First run: just store hidden states and process normally
        if prev_hidden_states is None:
            self.cache_context.prev_hidden_states = hidden_states.detach().clone()
            return hidden_states, None, hidden_states, encoder_hidden_states
        
        # 计算token显著性，用于空间token减少
        motion_saliency = self.compute_motion_saliency(hidden_states)
        
        # 基于阈值将token分为静态和运动两类
        # M_t = {i : S_t^(i) > τ_s}, X_t^m = X_t[M_t], X_t^s = X_t[M_t^c]
        self.cache_context.static_token_mask = motion_saliency <= self.motion_threshold
        
        # Update cached states for next iteration
        self.cache_context.prev_hidden_states = hidden_states.detach().clone()
        
        return hidden_states, prev_hidden_states, hidden_states, encoder_hidden_states
    
    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        """Override to implement space-time FastCache with optional AdaCorrection and Token Merging"""
        # 如果启用了Token Merging，使用多阶段金字塔编码（在enhanced_process_blocks中处理）
        if self.enable_token_merge:
            return self.enhanced_process_blocks(start_idx, hidden, encoder, *args, **kwargs)
        
        # 如果启用了增强线性近似算法，使用增强版本
        if self.enable_enhanced_linear_approx:
            return self.enhanced_process_blocks(start_idx, hidden, encoder, *args, **kwargs)
        
        # 如果启用AdaCorrection，使用AdaCorrection处理流程
        if self.enable_adacorrection:
            return self.process_transformer_blocks(start_idx, hidden, encoder, *args, **kwargs)
        
        # 如果使用transformer级缓存，直接使用线性投影
        if self.use_cache:
            # H_{t,l} = W_l H_{t,l-1} + b_l (线性近似)
            return self.block_projections[0](hidden), encoder
        
        # 空间Token减少：检查是否可以对部分token使用spatial减少
        static_mask = self.cache_context.static_token_mask
        if static_mask is not None and static_mask.any() and not static_mask.all():
            batch_size, seq_len, hidden_dim = hidden.shape
            
            # 将token分为motion和static两部分
            motion_indices = torch.where(~static_mask)[0]
            static_indices = torch.where(static_mask)[0]
            
            if len(motion_indices) > 0:
                # 获取运动token
                motion_hidden = hidden.index_select(1, motion_indices)
                motion_encoder = encoder.index_select(1, motion_indices) if encoder is not None else None
                
                # 通过完整的transformer块处理运动token
                processed_motion_hidden, processed_motion_encoder = self.process_transformer_blocks(
                    start_idx, motion_hidden, motion_encoder, *args, **kwargs
                )
                
                # 使用线性投影处理静态token: H_t^s = W_c X_t^s + b_c
                static_hidden = hidden.index_select(1, static_indices)
                static_encoder = encoder.index_select(1, static_indices) if encoder is not None else None
                static_hidden = self.spatial_projection(static_hidden)
                
                # 合并结果
                result_hidden = hidden.clone()
                result_hidden.index_copy_(1, motion_indices, processed_motion_hidden)
                result_hidden.index_copy_(1, static_indices, static_hidden)
                
                result_encoder = encoder.clone() if encoder is not None else None
                if result_encoder is not None:
                    result_encoder.index_copy_(1, motion_indices, processed_motion_encoder)
                    result_encoder.index_copy_(1, static_indices, static_encoder)
                
                return result_hidden, result_encoder
        
        # 如果没有空间token减少，则走正常的transformer处理流程
        return self.process_transformer_blocks(start_idx, hidden, encoder, *args, **kwargs)
    
    def process_transformer_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        """Process hidden states through transformer blocks with per-block caching and optional AdaCorrection"""
        current_hidden, current_encoder = hidden, encoder
        prev_hidden = self.cache_context.prev_hidden_states
        
        # 对每个transformer块分别决定是否使用缓存
        for i, block in enumerate(self.transformer_blocks[start_idx:], start=start_idx):
            # 如果启用AdaCorrection，使用自适应混合策略
            if self.enable_adacorrection and prev_hidden is not None:
                # 计算cached路径（线性近似）
                cached_hidden = self.block_projections[i](current_hidden)
                
                # 计算fresh路径（完整transformer块）
                fresh_hidden, fresh_encoder = block(current_hidden, current_encoder, *args, **kwargs)
                fresh_hidden, fresh_encoder = (fresh_hidden, fresh_encoder) if self.return_hidden_states_first else (fresh_encoder, fresh_hidden)
                
                # 计算offset score S_t^l
                offset_score = self.compute_adacorr_offset_score(current_hidden, prev_hidden)
                
                # 计算correction weight λ_t^l = clip(γ * S_t^l, 0, 1)
                correction_weight = torch.clamp(self.adacorr_gamma * offset_score, 0.0, 1.0)
                
                # 自适应混合: ĥ_{t,l+1} = (1 - λ_t^l) * h̃_{t,l+1} + λ_t^l * h_{t,l+1}
                # correction_weight is a scalar, so we can directly use it for blending
                current_hidden = (1 - correction_weight) * cached_hidden + correction_weight * fresh_hidden
                current_encoder = fresh_encoder
                
                # 更新prev_hidden为当前层的输出，用于下一层的计算
                prev_hidden = current_hidden.detach().clone()
                
            # 如果未启用AdaCorrection，使用原有的FastCache逻辑
            elif prev_hidden is not None:
                # 计算相对变化
                delta = self.compute_relative_change(current_hidden, prev_hidden)
                
                # 基于统计检验决定是否使用线性近似（可学习缓存）
                if delta <= self.rel_l1_thresh:
                    # 使用线性投影近似
                    current_hidden = self.block_projections[i](current_hidden)
                    self.cache_hits += 1
                    # 更新prev_hidden
                    prev_hidden = current_hidden.detach().clone()
                    continue
            
            # 完整执行transformer处理（当不使用缓存时）
            if not (self.enable_adacorrection and prev_hidden is not None):
                current_hidden, current_encoder = block(current_hidden, current_encoder, *args, **kwargs)
                current_hidden, current_encoder = (current_hidden, current_encoder) if self.return_hidden_states_first else (current_encoder, current_hidden)
                prev_hidden = current_hidden.detach().clone()
        
        # 处理single_transformer_blocks如果存在
        if self.single_transformer_blocks:
            current_hidden = torch.cat([current_encoder, current_hidden], dim=1)
            for block in self.single_transformer_blocks:
                current_hidden = block(current_hidden, *args, **kwargs)
            current_encoder, current_hidden = current_hidden.split([current_encoder.shape[1], current_hidden.shape[1] - current_encoder.shape[1]], dim=1)
        
        return current_hidden, current_encoder
    
    def compute_relative_change(self, current, previous):
        """计算当前和上一时间步隐藏状态的相对变化"""
        if previous is None:
            return float('inf')
            
        # 计算Frobenius范数
        diff_norm = torch.norm(current - previous, p='fro')
        prev_norm = torch.norm(previous, p='fro')
        
        # 避免除以零
        if prev_norm == 0:
            return float('inf')
            
        return (diff_norm / prev_norm).item()
