# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import triton
import triton.language as tl

from fla.ops.utils.op import exp, log, make_tensor_descriptor


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=['S']
)
@triton.jit(do_not_specialize=['T'])
def logcumsumexp_fwd_kernel(
    s,
    z,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_mp = tl.full([S,], float('-inf'), dtype=tl.float32)
    b_zp = tl.zeros([S,], dtype=tl.float32)

    if USE_TMA:
        desc_s = make_tensor_descriptor(s + i_bh * T * S,
                                        shape=[T, S], strides=[S, 1], block_shape=[BT, S])
        desc_z = make_tensor_descriptor(z + i_bh * T * S,
                                        shape=[T, S], strides=[S, 1], block_shape=[BT, S])

    for i_t in range(tl.cdiv(T, BT)):
        if USE_TMA:
            b_s = desc_s.load([i_t * BT, 0]).to(tl.float32)
        else:
            p_s = tl.make_block_ptr(s + i_bh * T*S, (T, S), (S, 1), (i_t * BT, 0), (BT, S), (1, 0))
            # [BT, S]
            b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

        # [S,]
        b_mc = tl.max(b_s, 0)
        b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * exp(b_mp - b_mc)
        # [BT, S]
        b_s = exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
        # [S,]
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        # [BT, BS]
        # small eps to prevent underflows
        b_z = log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
        if USE_TMA:
            desc_z.store([i_t * BT, 0], b_z.to(desc_z.dtype))
        else:
            p_z = tl.make_block_ptr(z + i_bh * T*S, (T, S), (S, 1), (i_t * BT, 0), (BT, S), (1, 0))
            tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))
