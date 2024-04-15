import random

import numpy as np
import torch

from ._semi_structured_conversions import sparse_semi_structured_from_dense_cutlass


def _get_perms_2_4():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        col_o = col // 2
        for block in [0, 1]:
            for row in [
                    2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col_o * 256 + 8 * (col % 2) +
                             4 * block)
        for j in range(4):
            perm.extend([p + 1 * j for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i * 8 + j for j in [0, 4, 1, 5, 2, 6, 3, 7]])
    scale_perm_single = []
    for i in range(8):
        scale_perm_single.extend([8 * i + j for j in [0, 1, 2, 3, 4, 5, 6, 7]])
    return perm, scale_perm, scale_perm_single


_perm_2_4, _scale_perm_2_4, _scale_perm_single_2_4 = _get_perms_2_4()


def unpack_gptq(w_gptq, size_k, size_n, num_bits):
    pack_factor = 32 // num_bits

    assert w_gptq.shape[0] * pack_factor == size_k
    assert w_gptq.shape[1] == size_n
    assert w_gptq.is_contiguous()

    res = np.zeros((size_k, size_n), dtype=np.uint32)
    w_gptq_cpu = w_gptq.cpu().numpy().astype(np.uint32)

    print("HERE!!!!!! w_gpq = {}".format(w_gptq_cpu))

    for i in range(pack_factor):
        vals = w_gptq_cpu & 0xf
        w_gptq_cpu >>= 4
        res[i::8, :] = vals

    res = torch.from_numpy(res.astype(np.int32)).to(w_gptq.device)

    print("HERE!!!!!! w_unpacked = {}".format(res[:128, 0]))

    return res


def reshape_to_group(w, s, size_k, size_n, group_size):
    assert w.shape[0] == size_k, "w.shape = {}, size_k/n = {}".format(
        w.shape, (size_k, size_n))
    assert w.shape[1] == size_n, "w.shape = {}, size_k/n = {}".format(
        w.shape, (size_k, size_n))
    assert w.dtype == torch.half or w.dtype == torch.int32

    if group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))
    s = s.reshape((1, -1))

    return w, s


def reshape_from_group(w, s, size_k, size_n, group_size):
    assert w.shape[0] == group_size, "w.shape[0] = {}, group_size = {}".format(
        w.shape[0], group_size)
    assert w.dtype == torch.half or w.dtype == torch.int32, "w.dtype = {}".format(
        w.dtype)

    if group_size < size_k:
        w = w.reshape((group_size, -1, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((size_k, size_n)).contiguous()

    s = s.reshape((-1, size_n)).contiguous()

    return w, s


def is_24_block(block):
    return (block != 0).sum() <= 2


def fix_24_block(block):
    # print("Fixing: block = {}".format(block))

    while not is_24_block(block):
        min_val = 100
        min_idx = 0
        for k in range(4):
            val = abs(block[k])
            if val != 0 and val < min_val:
                min_val = val
                min_idx = k
        block[min_idx] = 0

    # print("        new_block = {}".format(block))
    return block

def w_fix_24(w):
    block_size = 4

    size_k, size_n = w.shape

    w = w.reshape((-1, block_size, size_n))
    w = w.permute(1, 0, 2)
    w = w.reshape((block_size, -1))

    mask = w != 0

    non_zeros = torch.sum(mask, 0, keepdim=True)[0]

    n_total = 0
    n_fixes = 0
    for i in range(non_zeros.shape[0]):
        n_total += 1
        if non_zeros[i] > 2:
            w[:,i] = fix_24_block(w[:,i].reshape(-1))
            n_fixes += 1


    w = w.reshape((block_size, -1, size_n))
    w = w.permute(1, 0, 2)
    w = w.reshape((size_k, size_n)).contiguous()

    print("{} blocks fixed.".format(n_fixes / n_total))

    return w


def dequant(w, s, size_k, size_n, num_bits, group_size):
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2

    # Reshape to [groupsize, -1]
    w, s = reshape_to_group(w, s, size_k, size_n, group_size)

    # Dequantize

    # print("YYY w = {} w.shape = {}".format(w[:64,26807], w.shape))
    # print("YYY half_q_val = {}".format(half_q_val))
    # print("YYY w - half_q_val = {}".format((w - half_q_val)[:64,26807]))
    # print("YYY (w - half_q_val).half() = {}".format(((w - half_q_val).half())[:64,26807]))
    # print("YYY (w - half_q_val).half() * s = {}".format((((w - half_q_val).half()) * s)[:64,26807]))

    # mask = w != half_q_val
    # print("MASK mask = {}".format(mask))

    w_norm = w - half_q_val
    # w_norm = w_fix_24(w_norm)

    res = w_norm.half() * s

    # print("RES RES res = {} type = {}".format(res, res.type()))
    # check_24(w - half_q_val)
    # Restore shapes
    res, s = reshape_from_group(res, s, size_k, size_n, group_size)

    return res


def compress_24(w):
    w = w.t().contiguous()
    w_comp, meta = sparse_semi_structured_from_dense_cutlass(w)
    w_comp = w_comp.t().contiguous()
    return w_comp, meta


def quant(w, s, size_k, size_n, num_bits, group_size):
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2

    # Reshape to [groupsize, -1]
    w, s = reshape_to_group(w, s, size_k, size_n, group_size)

    # Quantize
    q_w = torch.round(w / s).int()
    q_w += half_q_val
    q_w = torch.clamp(q_w, 0, max_q_val)

    # Restore shapes
    q_w, s = reshape_from_group(q_w, s, size_k, size_n, group_size)

    return q_w


def check_24(w, num_rows_to_sample=50, _verbose=False):
    BLOCK_SIZE = 4
    MAX_NON_ZEROS = 2

    w = w.t().contiguous()

    print("check_24: w.shape = {}".format(w.shape))

    num_rows, num_cols = w.shape
    sampled_row_idxs = random.choices(range(num_rows), k=num_rows_to_sample)
    if _verbose:
        print(f"Sampled row idxs = {sampled_row_idxs}")

    total_segments = 0
    non_24_segments = 0
    for i in sampled_row_idxs:
        for j in range(0, num_cols - BLOCK_SIZE, BLOCK_SIZE):
            total_segments += 1
            block = w[i, j:j + BLOCK_SIZE]
            num_nonzero = torch.count_nonzero(block)
            if num_nonzero > MAX_NON_ZEROS:
                print("i = {} j = {} block = {}".format(i, j, block))
                non_24_segments += 1

    print(f"{non_24_segments} / {total_segments} do not have 2:4 structure.")


def repack_gptq_to_marlin_24(w_gptq,
                             scales,
                             size_k,
                             size_n,
                             num_bits,
                             group_size,
                             marlin_tile=16):
    assert num_bits == 4

    if group_size == -1:
        group_size = size_k

    print(
        "repack_gptq_to_marlin_24:\n    num_bits = {}\n    group_size = {}\n    w_gptq.shape = {}\n    size_k/n = {}"
        .format(num_bits, group_size, w_gptq.shape, (size_k, size_n)))

    # Dequantize
    w_unpacked = unpack_gptq(w_gptq, size_k, size_n, num_bits)
    print("    w_unpacked: shape = {} type = {}".format(
        w_unpacked.shape, w_unpacked.type()))

    w = dequant(w_unpacked, scales, size_k, size_n, num_bits, group_size)
    print("    w dequant: shape = {} type = {}".format(w.shape, w.type()))

    # check_24(w)

    # Compress
    w_comp, meta = compress_24(w)
    print("    w_comp: shape = {} type = {}".format(w_comp.shape,
                                                    w_comp.type()))
    print("    meta:   shape = {} type = {}".format(meta.shape, meta.type()))

    size_k_comp = size_k // 2
    group_size_comp = group_size // 2

    # Quantize the compressed weight
    q_w = quant(w_comp, scales, size_k_comp, size_n, num_bits, group_size_comp)
    print("    q_w: shape = {} type = {}".format(q_w.shape, q_w.type()))

    print("HERE: q_w = {}".format(q_w))
    # time.sleep(300)
    # Reshuffle to marlin_24 format
    q_w = q_w.reshape((size_k_comp // marlin_tile, marlin_tile,
                       size_n // marlin_tile, marlin_tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k_comp // marlin_tile, size_n * marlin_tile))

    res = q_w
    res = res.reshape((-1, _perm_2_4.numel()))[:, _perm_2_4].reshape(res.shape)

    # Pack
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res[:, i::8] << 4 * i

    q = torch.from_numpy(q.astype(np.int32)).to(w_gptq.device)
    print("    q: shape = {} type = {}".format(q.shape, q.type()))

    return q, meta, w


def repack_scales_to_marlin_24(s, group_size, size_k, size_n):
    if group_size == -1:
        group_size = size_k

    if group_size != size_k:
        # s = s.reshape((1, -1))  # TODO: Remove this line after testing
        s = s.reshape((-1, len(_scale_perm_2_4)))[:, _scale_perm_2_4]
    else:
        s = s.reshape(
            (-1, len(_scale_perm_single_2_4)))[:, _scale_perm_single_2_4]
    s = s.reshape((-1, size_n)).contiguous()

    return s
