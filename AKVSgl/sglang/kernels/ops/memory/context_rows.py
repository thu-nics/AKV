"""Decode slot scatter for request-owned overflowing raw histories."""

import triton
import triton.language as tl


@triton.jit
def write_row_slots(Pointers, Rows, Columns, Values, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    valid = i < N
    row = tl.load(Rows + i, valid, 0)
    column = tl.load(Columns + i, valid, 0)
    pointer = tl.load(Pointers + row, valid, 0).to(tl.pointer_type(tl.int32))
    value = tl.load(Values + i, valid, 0)
    tl.store(pointer + column, value, valid)
