"""Decode read indices over native raw KV rows, without moving cached K/V."""

import triton
import triton.language as tl


@triton.jit
def context_window_lengths(Registry, Rows, Lengths, Output, WINDOW: tl.constexpr):
    b = tl.program_id(0)
    row = tl.load(Rows + b)
    length = tl.load(Lengths + b)
    entry = Registry + row * 6
    pointer = tl.load(entry)
    result = tl.minimum(length, WINDOW)
    if pointer != 0:
        positions = tl.load(entry + 1).to(tl.pointer_type(tl.int32))
        active = tl.load(entry + 3).to(tl.int32)
        next_pos = tl.load(entry + 4).to(tl.int32)
        generated = length - active
        # Native RadixAttention's window is the inclusive distance q_pos-k_pos.
        lower = next_pos + generated - 1 - WINDOW
        lo = tl.full((), 0, tl.int32)
        hi = active
        while lo < hi:
            mid = (lo + hi) // 2
            pos = tl.load(positions + mid)
            lo = tl.where(pos < lower, mid + 1, lo)
            hi = tl.where(pos < lower, hi, mid)
        skipped_outputs = tl.maximum(0, lower - next_pos)
        result = active - lo + generated - skipped_outputs
    tl.store(Output + b, result)


@triton.jit
def context_decode_indices(
    Registry,
    Table,
    Rows,
    Lengths,
    Indptr,
    Output,
    Starts,
    Mapping,
    ROW_STRIDE: tl.constexpr,
    TRANSLATE: tl.constexpr,
    MULTIPLIER: tl.constexpr,
    HAS_START: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    b = tl.program_id(0)
    row = tl.load(Rows + b)
    length = tl.load(Lengths + b)
    offset = tl.load(Indptr + b)
    start = tl.load(Starts + b) if HAS_START else 0
    entry = Registry + row * 6
    pointer = tl.load(entry)
    active = tl.load(entry + 3).to(tl.int32)
    prompt = tl.load(entry + 2).to(tl.int32)
    raw_ids = pointer.to(tl.pointer_type(tl.int32))
    row_pointer = tl.load(entry + 5)
    raw_row = Table + row * ROW_STRIDE
    if row_pointer != 0:
        raw_row = row_pointer.to(tl.pointer_type(tl.int32))
    for block in range(tl.cdiv(length, BLOCK)):
        i = block * BLOCK + tl.arange(0, BLOCK)
        valid = i < length
        logical = start + i
        raw = logical
        if pointer != 0:
            in_prompt = logical < active
            raw = tl.load(raw_ids + logical, mask=valid & in_prompt, other=0)
            raw = tl.where(in_prompt, raw, prompt + logical - active)
        slot = tl.load(raw_row + raw, mask=valid, other=0)
        if TRANSLATE:
            slot = tl.load(Mapping + slot, mask=valid, other=0) * MULTIPLIER
        tl.store(Output + offset + i, slot, mask=valid)
