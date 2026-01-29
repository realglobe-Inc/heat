# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------


import torch

try:
    import MultiScaleDeformableAttention as msda

    _import_error = None
except ImportError as e:
    msda = None
    _import_error = str(e)


from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import functional


def is_extension_available() -> bool:
    """
    Check if the C++ extension (MultiScaleDeformableAttention) is available.

    :returns: True if the extension is loaded, False otherwise.
    """
    return msda is not None


def get_extension_error() -> str | None:
    """
    Get the error message if the C++ extension failed to load.

    :returns: Error message or None if it was loaded successfully.
    """
    return _import_error


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        if msda is not None and value.is_cuda:
            output = msda.ms_deform_attn_forward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                ctx.im2col_step,
            )
        else:
            output = ms_deform_attn_core_pytorch(
                value, value_spatial_shapes, sampling_locations, attention_weights
            )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors

        if msda is not None and value.is_cuda:
            grad_value, grad_sampling_loc, grad_attn_weight = (
                msda.ms_deform_attn_backward(
                    value,
                    value_spatial_shapes,
                    value_level_start_index,
                    sampling_locations,
                    attention_weights,
                    grad_output,
                    ctx.im2col_step,
                )
            )
        else:
            # Fallback to autograd for CPU/missing extension
            # Since forward was done with pure pytorch, we can actually
            # just not use MSDeformAttnFunction for that case.
            # But here we are already inside a Function.
            # To avoid complexity, we just throw an error if we reach here without MSDA
            # but wait, if it was done with pytorch implementation in forward,
            # then autograd would have handled it IF we didn't wrap it in a Function.
            raise NotImplementedError(
                "Backward for MSDeformAttn on CPU or without extension is not implemented via MSDeformAttnFunction. "
                "Use the pytorch implementation directly for autograd support."
            )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    # for debug and test only,
    # need to use cuda version instead
    n_, s_, m_, d_ = value.shape
    _, lq_, m_, l_, p_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # n_, H_*W_, m_, d_ -> n_, H_*W_, m_*d_ -> n_, m_*d_, H_*W_ -> n_*m_, d_, H_, W_
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(n_ * m_, d_, H_, W_)
        )
        # n_, lq_, m_, p_, 2 -> n_, m_, lq_, p_, 2 -> n_*m_, lq_, p_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # n_*m_, d_, lq_, p_
        sampling_value_l_ = functional.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (n_, lq_, m_, l_, p_) -> (n_, m_, lq_, l_, p_) -> (n_, m_, 1, lq_, l_*p_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        n_ * m_, 1, lq_, l_ * p_
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(n_, m_ * d_, lq_)
    )
    return output.transpose(1, 2).contiguous()
