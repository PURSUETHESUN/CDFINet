import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
import pywt
from torch.autograd import Function
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# triton cross scan, 2x speed than pytorch implementation =========================
import torch
import triton
import triton.language as tl

@triton.jit
def triton_cross_scan(
        x,  # (B, C, H, W)
        y,  # (B, 4, C, H, W)
        BC: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
        DC: tl.constexpr,
        DH: tl.constexpr,
        DW: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = _tmp0 + i_h * BH * DW + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2  # same
    p_y2 = y + i_b * 4 * _tmp1 + _tmp1 + _tmp0 + i_w * BW * DH + tl.arange(0, BW)[None, :] * DH + i_h * BH + tl.arange(
        0, BH)[:, None]  # trans
    p_y3 = y + i_b * 4 * _tmp1 + 2 * _tmp1 + _tmp0 + (NH - i_h - 1) * BH * DW + (
                BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (
                       BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW)  # flip
    p_y4 = y + i_b * 4 * _tmp1 + 3 * _tmp1 + _tmp0 + (NW - i_w - 1) * BW * DH + (
                BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1) * BH + (
                       BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW - NW * BW) * DH  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        tl.store(p_y2 + _idx, _x, mask=_mask_hw)
        tl.store(p_y3 + _idx, _x, mask=_mask_hw)
        tl.store(p_y4 + _idx, _x, mask=_mask_hw)


@triton.jit
def triton_cross_merge(
        x,  # (B, C, H, W)
        y,  # (B, 4, C, H, W)
        BC: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
        DC: tl.constexpr,
        DH: tl.constexpr,
        DW: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = _tmp0 + i_h * BH * DW + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2  # same
    p_y2 = y + i_b * 4 * _tmp1 + _tmp1 + _tmp0 + i_w * BW * DH + tl.arange(0, BW)[None, :] * DH + i_h * BH + tl.arange(
        0, BH)[:, None]  # trans
    p_y3 = y + i_b * 4 * _tmp1 + 2 * _tmp1 + _tmp0 + (NH - i_h - 1) * BH * DW + (
                BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (
                       BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW)  # flip
    p_y4 = y + i_b * 4 * _tmp1 + 3 * _tmp1 + _tmp0 + (NW - i_w - 1) * BW * DH + (
                BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1) * BH + (
                       BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW - NW * BW) * DH  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y1 = tl.load(p_y1 + _idx, mask=_mask_hw)
        _y2 = tl.load(p_y2 + _idx, mask=_mask_hw)
        _y3 = tl.load(p_y3 + _idx, mask=_mask_hw)
        _y4 = tl.load(p_y4 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

#V4
class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(
            triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y.view(B, 4, C, -1)

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x

#V4
class CrossMergeTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor):
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(
            triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x.view(B, C, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
#V01
class SelectiveScanMamba(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        # all in float
        # if u.stride(-1) != 1:
        #     u = u.contiguous()
        # if delta.stride(-1) != 1:
        #     delta = delta.contiguous()
        # if D is not None and D.stride(-1) != 1:
        #     D = D.contiguous()
        # if B.stride(-1) != 1:
        #     B = B.contiguous()
        # if C.stride(-1) != 1:
        #     C = C.contiguous()
        # if B.dim() == 3:
        #     B = B.unsqueeze(dim=1)
        #     ctx.squeeze_B = True
        # if C.dim() == 3:
        #     C = C.unsqueeze(dim=1)
        #     ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        # dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        # dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

#V2
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

#V4 etl.
class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

#Note: we did not use csm_triton in and before vssm1_0230, we used pytorch version !
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs

def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    if no_einsum:
        x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                         bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
    else:
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


# =====================================================
class GRN_T(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1,dim,1,1))
        self.beta = nn.Parameter(torch.zeros(1,dim,1,1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=True), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sharedMLP(self.avg_pool(x) + self.max_pool(x))
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(3,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        minout, _ = torch.min(x, dim=1,keepdim=True)
        x = torch.cat([avgout, maxout, minout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args).contiguous()

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class DFS(nn.Module):
    def __init__(self,dim: int = 0 ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=dim,out_channels=dim//2,kernel_size=1,bias=True),
            nn.GELU()
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=dim,out_channels=dim//2,kernel_size=1,bias=True),
            nn.GELU()
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1,bias=True),
            nn.Sigmoid()
        )


        self.sp = nn.Sequential(
            nn.Conv2d(in_channels=dim*2,out_channels=dim*2,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, stride=1,padding=1,groups=dim*2),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3,padding=1, groups=dim, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )



    def forward(self, inp1, inp2):
        x1 = self.c1(self.avg_pool(inp1))
        x2 = self.c2(self.avg_pool(inp2))
        x_c = torch.cat([x1,x2], dim=1)
        x_c = self.c3(x_c)
        #print("x_c",x_c.size())

        x_s = torch.cat([inp1,inp2], dim=1)
        x_s = self.sp(x_s)
        #print("x_s", x_s.size())

        xd = x_c * x_s
        xd = self.dw_conv(xd)
        #print("x_a", x_a.size())
        x1 = inp1 * xd
        x2 = inp2 * xd

        return x1, x2, xd

class GFDS(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            # ======================
            k_size = 9,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        dwdim = d_model
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.out_norm = nn.LayerNorm(d_inner)
        k_size = k_size

        self.forward_core1 = partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                       CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton)
        self.forward_core2 = partial(self.forward_corev22, force_fp32=True, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                       CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton)

        k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        self.dwt = DWT_2D(wave='haar')
        self.dwconv_ll1 = nn.Sequential(
            nn.Conv2d(dwdim, dwdim, kernel_size=3, stride=1, dilation=1, padding=1, groups=dwdim),
            nn.SiLU(),
        )
        self.dwconv_lh1 = nn.Sequential(
            nn.Conv2d(dwdim, dwdim, kernel_size=(1, k_size), padding=(0, k_size // 2), groups=dwdim),
            nn.SiLU(),
        )
        self.dwconv_hl1 = nn.Sequential(
            nn.Conv2d(dwdim, dwdim, kernel_size=(k_size, 1), padding=(k_size // 2, 0), groups=dwdim),
            nn.SiLU(),
        )
        self.dwconv_hh1 = nn.Sequential(
            nn.Conv2d(dwdim, dwdim, kernel_size=3, stride=1, dilation=1, padding=1, groups=dwdim),
            nn.SiLU(),
        )

        self.split_wave = (dwdim, dwdim, dwdim, dwdim)

        self.dwconv_ll2 = nn.Sequential(
            nn.Conv2d(dwdim, dwdim, kernel_size=3, stride=1, padding=1, groups=dwdim),
            nn.SiLU(),
        )
        self.dwconv_lh2 = nn.Sequential(
            nn.Conv2d(dwdim, dwdim, kernel_size=(k_size, 1), padding=(k_size // 2, 0), groups=dwdim),
            nn.SiLU(),
        )
        self.dwconv_hl2 = nn.Sequential(
            nn.Conv2d(dwdim, dwdim, kernel_size=(1, k_size), padding=(0, k_size // 2), groups=dwdim),
            nn.SiLU(),
        )
        self.dwconv_hh2 = nn.Sequential(
            nn.Conv2d(dwdim, dwdim, kernel_size=3, stride=1, padding=1, groups=dwdim),
            nn.SiLU(),
        )

        self.dw = nn.Sequential(
            nn.Conv2d(in_channels=4*dwdim,out_channels=4*dwdim,kernel_size=3,stride=1,padding=1,groups=4*dwdim),
            nn.SiLU(),
        )

        self.s_conv = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model,kernel_size=3,padding=1,stride=1,groups=d_model),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )


        self.f_conv = nn.Sequential(
            nn.Conv2d(in_channels=4*dwdim, out_channels=1,kernel_size=1,stride=1),
            nn.Sigmoid()
        )

        self.idwt = IDWT_2D(wave='haar')


        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        self.x_proj_weight2 = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Linear(d_model*2, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

            #dt proj2=======================
            self.dt_projs2 = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight2 = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs2], dim=0))  # (K, inner, rank)
            self.dt_projs_bias2 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs2], dim=0))  # (K, inner)
            del self.dt_projs2

            # A, D =======================================
            self.A_logs2 = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds2 = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scan,
                       force_fp32=None, no_einsum=False, CrossScan=CrossScan, CrossMerge=CrossMerge):
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=no_einsum,
        )
        return x

    def forward_corev22(self, x: torch.Tensor, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scan,
                       force_fp32=None, no_einsum=False, CrossScan=CrossScan, CrossMerge=CrossMerge):
        x = cross_selective_scan(
            x, self.x_proj_weight2, None, self.dt_projs_weight2, self.dt_projs_bias2,
            self.A_logs2, self.Ds2, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=no_einsum,
        )
        return x

    def forward(self, x: torch.Tensor, **kwargs): #B, H, W, C

        x = self.in_proj(x) ##B, H, W, C -> B, H, W, 2*C

        x, z = x.chunk(2, dim=-1)  # (b, h, w, C)

        z = self.act(z)  # B, H, W, 4C -> B, H, W, 4C

        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        x_s = self.s_conv(x)

        x_dwt = self.dwt(x)  # B, C, H, W -> B, 4C, H, W
        x_ll1, x_lh1, x_hl1, x_hh1 = torch.split(x_dwt, self.split_wave, dim=1)
        x = torch.cat((self.dwconv_ll1(x_ll1), self.dwconv_lh1(x_lh1),
                       self.dwconv_hl1(x_hl1), self.dwconv_hh1(x_hh1)),dim=1)
        x_ll2, x_lh2, x_hl2, x_hh2 = torch.split(x, self.split_wave, dim=1)
        x = torch.cat((self.dwconv_ll2(x_ll2), self.dwconv_lh2(x_lh2),
                       self.dwconv_hl2(x_hl2), self.dwconv_hh2(x_hh2)),dim=1) + x

#**************Cross_channel selection*****************
        x = x.view(B, 4, C, H//2, W//2)
        # transpose 1, 2 axis
        x = x.transpose(1, 2).contiguous()
        x = x.view(B,-1, H//2, W//2)
        x1, x2 = torch.chunk(x,2,dim=1)

        y1 = self.forward_core1(x1) #SS2D
        y2 = self.forward_core2(x2) #SS2D
        y = torch.cat((y1,y2),dim=-1)

        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.dw(y)
        x_f = self.f_conv(y)
        x_f = F.interpolate(input=x_f, size=z.size()[1:3], mode="bilinear", align_corners=False)

        y = self.idwt(y)  # B, 4C, H, W -> B, C, H, W
        y = y.permute(0, 2, 3, 1).contiguous()  # B, C, H, W -> B, H, W, C
        y = y * z  # B, H, W, C -> B, H, W, C

#********Gated spectral attention*********
        x_s = x_s * x_f
        x_s = x_s.permute(0, 2, 3, 1).contiguous()

        out = self.dropout(self.out_proj(torch.cat([y, x_s],dim=-1))) # B, H, W, C -> B, H, W, C

        return out

class LFDS(nn.Module):
    def __init__(self, inp, oup,kernel_size = 9, expansion=1,branch_ratio=0.125):
        super().__init__()
        kernel_size = kernel_size
        hidden_dim = int(oup * expansion)
        dwdim = hidden_dim

        self.inp = nn.Sequential(
            nn.Conv2d(in_channels=inp,out_channels=hidden_dim,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.dwt = DWT_2D(wave='haar')

        self.dwconv_ll = nn.Conv2d(dwdim, dwdim, kernel_size=3, stride=1, padding=1, groups=dwdim)

        self.dwconv_lh = nn.Conv2d(dwdim, dwdim, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2),
                                   groups=dwdim)
        self.dwconv_hl = nn.Conv2d(dwdim, dwdim, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0),
                                   groups=dwdim)
        self.dwconv_hh = nn.Conv2d(dwdim, dwdim, kernel_size=3, stride=1, padding=1, groups=dwdim)

        self.split_wave = (dwdim, dwdim, dwdim, dwdim)

        self.split_h = (dwdim, 3 * dwdim)
        self.fuse_H = nn.Sequential(
            nn.Conv2d(in_channels=3 * dwdim, out_channels=3 * dwdim, kernel_size=1, stride=1),
            nn.BatchNorm2d(3 * dwdim),
            nn.GELU(),
            nn.Conv2d(in_channels=3 * dwdim, out_channels=3 * dwdim, kernel_size=3, stride=1, padding=1,
                      groups=3 * dwdim),
            nn.GELU(),
        )
        self.act = nn.GELU()

        self.idwt = IDWT_2D(wave='haar')

        self.s_conv = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,kernel_size=3,padding=1,stride=1,groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )


        self.f_conv = nn.Sequential(
            nn.Conv2d(in_channels=3*dwdim, out_channels=1,kernel_size=1,stride=1),
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim*2,out_channels=oup,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(oup),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.inp(x)
        x_s = self.s_conv(x)

        x_dwt = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = torch.split(x_dwt, self.split_wave, dim=1)
        x_dwt = self.act(torch.cat((self.dwconv_ll(x_ll), self.dwconv_lh(x_lh), self.dwconv_hl(x_hl), self.dwconv_hh(x_hh)),
                          dim=1))
        xl, xh = torch.split(x_dwt,self.split_h,dim=1)
        xh = self.fuse_H(xh)
        x_c = self.f_conv(xh)
        x_c = F.interpolate(input=x_c, size=x.size()[2:], mode="bilinear", align_corners=False)

        x_idwt = self.idwt(torch.cat([xl, xh], dim=1))
# ********Gated spectral attention*********
        x_s = x_c * x_s
        x = torch.cat([x_s, x_idwt],dim=1)

        x = self.out(x)

        return x

class MFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=-1.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        dwdim = hidden_features//3 or in_features

        self.fc0 = nn.Sequential(
            nn.Conv1d(in_channels=in_features,out_channels=hidden_features,kernel_size=1,stride=1),
            nn.BatchNorm1d(hidden_features),
            nn.GELU()
        )

        for i in range(2):
            self.add_module(f'dw{str(i)}',
                            nn.Sequential(nn.Conv1d(dwdim, dwdim, kernel_size=3, stride=1, padding=1+i, dilation=i+1, groups=dwdim),
                                          nn.GELU()))
        self.add_module(f'dw{str(2)}',
                        nn.Sequential(nn.Conv1d(dwdim, dwdim, kernel_size=5, stride=1, padding=4, dilation=2,groups=dwdim),
                                      nn.GELU()))

        self.gate = nn.Sequential(
            nn.Conv1d(in_channels=hidden_features,out_channels=1,kernel_size=1,stride=1),
            nn.Sigmoid()
        )

        self.dw = nn.Sequential(
            nn.Conv1d(hidden_features,hidden_features,kernel_size=3,stride=1,padding=1,groups=hidden_features,bias=False),
            nn.GELU(),
            GRN_T(hidden_features),
        )

        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_features,out_channels=out_features,kernel_size=1,stride=1),
            nn.BatchNorm1d(out_features),
            nn.GELU()
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc0(x)
        #B, C, H, W = x.size()

        #gate branch
        x_gate = self.gate(x)

        featureout = []
        featurein = x.chunk(3, dim=1)
        for i in range(3):
            feat = getattr(self, f'dw{str(i)}')(featurein[i])
            if i > -1:
                feat = feat + featureout[-2]
            featureout.append(feat)
        x = torch.cat(featureout, dim=0)

        x = x_gate * x + x
        x = self.dw(x)
        x = self.fc1(x)
        x = self.drop(x)
        return x

class LGFS(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            k_size = 9,
            **kwargs,
    ):
        super().__init__()
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

#************LFDS block************
        self.lfds = LFDS(inp=hidden_dim,oup=hidden_dim,kernel_size=k_size, expansion=1)

# ************GFDS block************
        self.norm = norm_layer(hidden_dim)
        ssm_ratio = ssm_ratio if ssm_ratio==2 else 2
        self.gfds = GFDS(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            # ==========================
            forward_type=forward_type,
            k_size = k_size,
        )

        self.dfs = DFS(dim=hidden_dim)

        self.DW = nn.Sequential(
              nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,stride=1,padding=1,groups=hidden_dim,bias=True),
              nn.GELU(),
              GRN_T(hidden_dim)
            )

        self.o_prj = nn.Sequential(
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mffn = MFFN(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):#B,C,H,W

        x1 = self.drop_path(self.lfds(input))

        x2 = input.permute(0, 2, 3, 1).contiguous()
        x2 = self.drop_path(self.gfds(self.norm(x2)))
        x2 = x2.permute(0, 3, 1, 2).contiguous()

        x1, x2, xd = self.dfs(x1, x2)
        x = self.DW(x1+x2+xd)
        x = input + self.o_prj(x)

        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.mffn(x))# MFFN
            else:
                x = x + self.drop_path(self.mffn(x))  # MFFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Attention(torch.nn.Module):
    r""" Cascaded Group Attention.
    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, num_heads=4, dim_head=32,
                 attn_ratio=4,
                 kernels=[1, 3, 5, 7], ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.key_dim = dim // num_heads
        self.scale = self.key_dim ** -0.5
        #self.d = int(attn_ratio * dim_head)
        self.attn_ratio = attn_ratio

        qkvs = []
        dwk = []
        dwv = []
        for i in range(num_heads):
            qkvs.append(nn.Sequential(
                nn.Conv2d(in_channels=dim // (num_heads), out_channels=self.key_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
            ))#默认1*1卷积
            dwk.append(nn.Conv2d(in_channels=self.key_dim, out_channels=self.key_dim, kernel_size=kernels[i], stride=1,
                          padding=kernels[i] // 2, groups=self.key_dim,bias=False))
            dwv.append(nn.Conv2d(in_channels=self.key_dim, out_channels=self.key_dim, kernel_size=kernels[i], stride=1,
                          padding=kernels[i] // 2, groups=self.key_dim,bias=False))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dwk = torch.nn.ModuleList(dwk)
        self.dwv = torch.nn.ModuleList(dwv)

        self.group_atten = nn.Sequential(
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=1,padding=1,groups=self.key_dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.conv = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1,stride=1,padding=0,bias=False)
        self.spa = SpatialAttention()
        self.c_qkv = nn.Conv2d(in_channels=dim,out_channels=3*dim,kernel_size=1,stride=1,padding=0,bias=False)

        self.dfs = DFS(dim=dim)

        self.dw = torch.nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            GRN_T(dim)
        )

        self.proj = torch.nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU())

    def forward(self, x):  # x (B,C,H,W)
        x = x.permute(0,3,1,2).contiguous()
        input = x
        B, C, H, W = x.shape

#**************MVSA********************
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.key_dim], dim=1)  # B, C/h, H, W
            k = self.dwk[i](k)
            v = self.dwv[i](v)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = attn.softmax(dim=-1)  # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.key_dim, H, W)  # BCHW
            feats_out.append(feat)
        x = torch.cat(feats_out, 1)

# **************SA********************
        x2 = self.conv(input)
        x2 = x2 * self.spa(x2)

        x1 = x * x2
# ************** CS + GC ********************
        x1 = x1.view(B, 4, C//4, H, W)
        # transpose 1, 2 axis
        x1 = x1.transpose(1, 2).contiguous()
        x1 = x1.view(B, C, H, W)
        x1 = self.group_atten(x1)

# **************GCSA********************
        x2 = x + x2
        x2 = self.c_qkv(x2)
        q, k, v = x2.chunk(3,dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C, N
        attn = (q @ k.transpose(-2, -1)) * ((H * W) ** -0.5)#BCC
        attn = attn.softmax(dim=-1)  # BCC
        x2 = (attn @ v).view(B, C, H, W) # B, C, H, W

        x1, x2, xd = self.dfs(x1,x2)

        x = self.dw(x1 + x2 + xd)

        x = self.proj(x)
        return x

class CFFN(nn.Module):
    def __init__(self, dim, hidden_dim,  dropout=0.):
        super().__init__()
        dwdim = hidden_dim//4
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,kernel_size=1,stride=1,padding=0),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim,out_channels=1,kernel_size=1,stride=1),
            nn.Sigmoid()
        )

        for i in range(4):
            self.add_module(f'dw{str(i)}',
                            nn.Sequential(
                                nn.Conv2d(dwdim, dwdim, kernel_size=3, stride=1, padding=1, dilation=1, groups=dwdim),
                                nn.GELU(),
                            ))
        self.dw = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=1,padding=1,groups=hidden_dim,bias=False),
            nn.GELU(),
            GRN_T(hidden_dim),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0,bias=True),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv1(x)

        # gate branch
        x_gate = self.gate(x)

        featureout = []
        featurein = x.chunk(4,dim=1)
        feat = featurein[0]
        for i in range(4):
            if i > 0 : # add the previous output to the input
                feat = feat + featurein[i]
            feat = getattr(self,f'dw{str(i)}')(feat)
            featureout.append(feat)
        x = torch.cat(featureout,dim=1)
        x = x * x_gate + x
        x = self.dw(x)
        x = self.conv2(x)

        return x

#MVBA blocks：rel-attention
class MVBA(nn.Module):
    def __init__(self, inp, oup,heads=4, dim_head=32, dropout=0.05):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.CPE = nn.Sequential(
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, groups=oup),
            nn.GELU(),
            GRN_T(oup),
        )

        self.attn = Attention(dim=oup)
        self.ff = CFFN(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Permute(0,2,3,1),
            PreNorm(oup, self.attn, nn.LayerNorm),
        )

        self.ffn = nn.Sequential(
            # Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.BatchNorm2d),
            # Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):#B,C,H,W

        x = x + self.CPE(x)
        x = x + self.attn(x)
        x = x + self.ffn(x)

        return x

class FSDA(nn.Module):
    def __init__(self, dim1, dim2, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim2 // num_heads
        self.scale = head_dim ** -0.5

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim1, dim1 // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim1 // 4),
            nn.GELU(),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
        )
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Conv2d(dim2, dim2, kernel_size=1,  stride=1, bias=False)
        self.to_kv = nn.Conv2d(dim2, dim2*2, kernel_size=1,  stride=1, bias=False)

        self.w_d = nn.Sequential(
            nn.Conv2d(dim2//4, dim2, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
        )
        self.c_a = ChannelAttention(in_planes=dim2)
        self.fc1 =  nn.Sequential(
            nn.Conv2d(dim2, dim2, kernel_size=1, padding=0, stride=1,bias=False),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
        )
        self.fc2 =  nn.Sequential(
            nn.Conv2d(dim2, dim2, kernel_size=1, padding=0, stride=1,bias=False),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
        )

        self.proj =  nn.Sequential(
            nn.Conv2d(dim2, dim2, kernel_size=1, padding=0, stride=1,bias=False),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x2.shape
        x_dwt = self.dwt(self.reduce(x1))#降维1/4+离散的小波变换升维原来的Channel
        x_dwt = self.filter(x_dwt)#3*3的卷积特征融合

        x_idwt = self.idwt(x_dwt)  # 小波变换逆操作进行捷径分支cat连接，通道数C//4
        x_idwt = self.w_d(x_idwt)
        x_idwt = self.c_a(x_idwt) * x_idwt
        q = self.to_q(x_dwt)
        q = q.reshape( B, -1 , C)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)

        kv = self.to_kv(x2).reshape(B, 2*C, -1).permute(0, 2, 1).contiguous().chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_heads), kv)#循环二次，对qk分头

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)
        x = rearrange(x, 'b (ih iw) c -> b c ih iw', ih=H, iw=W)
        # print("x", x.size())
        # print("x_idwt", x_idwt.size())
        x = self.fc1(x + x_idwt)
        x = self.fc2(x_idwt) * x
        x = self.proj(x) + x2

        return x

class Downsample(nn.Module):
    def __init__(self,in_channels:int = 0, out_channels: int = 0):
        super().__init__()
        mid_channel = out_channels // 2
        dwdim = mid_channel // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.max = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.max(x)

        return x1 + x2

class CDFI(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[1, 1, 2, 1],
            dims=[64, 128, 192, 256],
            k_size=[9, 9, 9, 9],
            # =========================
            ssm_d_state=16,
            ssm_ratio=4.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v4",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # =========================
            drop_path_rate=0.0,
            patch_norm=True,
            norm_layer="LN",
            downsample_version: str = "v3",  # "v1", "v2", "v3"
            patchembed_version: str = "v2",  # "v1", "v2"
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        alpha = 2 #set LGFS stage

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        #
        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.Stem = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = Downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1]
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            if i_layer < alpha:
                self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                k_size = k_size[i_layer],))
            else:
                self.layers.append(self._make_layer_MVBA(
                inp = self.dims[i_layer],
                oup = self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=downsample,
                ))

        self.fsda = FSDA(dim1 = dims[-2], dim2 = dims[-1], num_heads=8)

        self.classifier = nn.Sequential(OrderedDict(
            avgpool = nn.AdaptiveAvgPool2d((1, 1)),
            flatten=nn.Flatten(1),
            ln = nn.LayerNorm(self.num_features, eps=1e-6),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    @staticmethod
    def _make_layer_MVBA(inp=96, oup=96,drop_path=[0.1,0.1],downsample = nn.Identity(),):
        depth = len(drop_path)
        layers = nn.ModuleList([])
        for i in range(depth):
            layers.append(MVBA(inp=inp, oup=oup))
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*layers, ),
            downsample=downsample,
        ))

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            k_size = 9,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(LGFS(
                    hidden_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    use_checkpoint=use_checkpoint,
                    k_size = k_size,
                ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.Stem(x)
        layer_outputs = []
        #print("layers",len(self.layers))
        for i, layer in enumerate(self.layers):
            if i == 2:
                x = layer.blocks(x)
                layer_outputs.append(x)
                x = layer.downsample(x)
            else:
                x = layer(x)
                layer_outputs.append(x)

        x3 = layer_outputs[-2]
        x4 = layer_outputs[-1]
        x = self.fsda(x3, x4)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    img_size = 224

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, img_size, img_size).to(device)


    model = CDFI(num_classes=8).to(device)
    print(model);
    clas = model(x)
    print(clas)
    print(clas.size())
    print(type(clas))
    # -- coding: utf-8 --
    from thop import profile

    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
