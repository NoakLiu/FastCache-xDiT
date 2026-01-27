"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import functools
import unittest

import torch
from torch import nn
from diffusers import DiffusionPipeline, FluxTransformer2DModel
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY

from xfuser.model_executor.cache import utils

def create_cached_transformer_blocks(use_cache, transformer, rel_l1_thresh, return_hidden_states_first, num_steps, motion_threshold=0.1, enable_enhanced_linear_approx=False, significance_level=0.05, enable_adacorrection=False, adacorr_gamma=1.0, adacorr_lambda=1.0, enable_token_merge=False, token_merge_k=5, token_merge_lambda=1.0, num_stages=3, merge_ratio=0.5):
    cached_transformer_class = {
        "Fb": utils.FBCachedTransformerBlocks,
        "Tea": utils.TeaCachedTransformerBlocks,
        "Fast": utils.FastCachedTransformerBlocks,
    }.get(use_cache)

    if not cached_transformer_class:
        raise ValueError(f"Unsupported use_cache value: {use_cache}")

    # FastCache requires motion_threshold parameter
    if use_cache == "Fast":
        return cached_transformer_class(
            transformer.transformer_blocks,
            transformer.single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=rel_l1_thresh,
            return_hidden_states_first=return_hidden_states_first,
            num_steps=num_steps,
            motion_threshold=motion_threshold,
            enable_enhanced_linear_approx=enable_enhanced_linear_approx,
            significance_level=significance_level,
            enable_adacorrection=enable_adacorrection,
            adacorr_gamma=adacorr_gamma,
            adacorr_lambda=adacorr_lambda,
            enable_token_merge=enable_token_merge,
            token_merge_k=token_merge_k,
            token_merge_lambda=token_merge_lambda,
            num_stages=num_stages,
            merge_ratio=merge_ratio,
            name=TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer)),
        )
    else:
        return cached_transformer_class(
            transformer.transformer_blocks,
            transformer.single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=rel_l1_thresh,
            return_hidden_states_first=return_hidden_states_first,
            num_steps=num_steps,
            name=TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer)),
        )


def apply_cache_on_transformer(
    transformer: FluxTransformer2DModel,
    *,
    rel_l1_thresh=0.12,
    return_hidden_states_first=False,
    num_steps=8,
    use_cache="Fb",
    motion_threshold=0.1,
    enable_enhanced_linear_approx=False,
    significance_level=0.05,
    enable_adacorrection=False,
    adacorr_gamma=1.0,
    adacorr_lambda=1.0,
    enable_token_merge=False,
    token_merge_k=5,
    token_merge_lambda=1.0,
    num_stages=3,
    merge_ratio=0.5,
):
    cached_transformer_blocks = nn.ModuleList([
        create_cached_transformer_blocks(
            use_cache, 
            transformer, 
            rel_l1_thresh, 
            return_hidden_states_first, 
            num_steps,
            motion_threshold,
            enable_enhanced_linear_approx,
            significance_level,
            enable_adacorrection,
            adacorr_gamma,
            adacorr_lambda,
            enable_token_merge,
            token_merge_k,
            token_merge_lambda,
            num_stages,
            merge_ratio,
        )
    ])

    dummy_single_transformer_blocks = torch.nn.ModuleList()

    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "transformer_blocks",
            cached_transformer_blocks,
        ), unittest.mock.patch.object(
            self,
            "single_transformer_blocks",
            dummy_single_transformer_blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    return transformer

# 
