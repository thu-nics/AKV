"""Per-request raw index storage, independent of model positions and KV size.

Only overflowing Context requests allocate a side row. Native graph-visible
request tables never resize; the side row has its final decode capacity at
admission and contains page IDs, not extra model KV.
"""

import torch


def context_publish_length(req, length):
    """Largest prefix whose absent terminal rows have an on-path Drop proof.

    A Drop at raw boundary b belongs to token b, so a b-token key does not
    contain it yet. Deferred active copies never authorize holes. Vectorized
    prefix maxima also handle an earlier hole whose proof lies past a later
    unpublishable row, without repeatedly rescanning a shrinking prefix.
    """
    state = getattr(req, "context_state", None)
    if state is None:
        return length
    import numpy as np

    rows = state.terminal_rows.numpy()[:length]
    missing = rows < 0
    if not missing.any():
        return length
    program = req.context_recompute_program or req.context_program
    proof_end = np.where(
        program.layout.keep_mask.numpy()[:len(rows)],
        length + 1,
        program.visible_until.numpy()[:len(rows)].astype(np.int64) + 1,
    )
    required = np.maximum.accumulate(np.where(missing, proof_end, 0))
    if required[-1] <= length:
        return length
    valid = np.flatnonzero(required <= np.arange(1, len(rows) + 1))
    return int(valid[-1] + 1) if len(valid) else 0


def needs_context_source_lease(req):
    """Keep a Retry branch only for nonterminal reads or unadopted cached gaps."""
    state = req.context_state
    canonical = state.canonical_rows
    present = canonical >= 0
    if bool((~state.owned[canonical[present]] & (
        canonical[present] != state.terminal_rows[present]
    )).any()):
        return True
    recovery = req.context_recovery_plan
    return recovery is not None and bool(
        recovery.reusable_prefix[len(canonical):].any()
    )


def prefill_progress_reserve(req, query_end):
    """Reserve singleton progress through future repair stages and final decode.

    Existing owners are already excluded from allocator availability. Future
    birth KV survives only to its Drop; intermediate copies coexist with those
    sources for one forward. Final-position copies are deferred until needed.
    Cache the CPU stage demand once per match, with no device page inspection.
    """
    import numpy as np

    program = req.context_recompute_program or req.context_program
    n = len(program.layout.positions)
    output = max(0, req.sampling_params.max_new_tokens - len(req.output_ids)) + 1
    window, plan = req.context_window_plan
    start = int(window.segment_query_starts[0])
    kept = plan.terminal_occurrences.numpy()
    births = window.birth_occurrences.numpy()[start:query_end]
    retained_birth = kept[start:query_end] == births
    if query_end < n:
        retained_birth |= program.visible_until.numpy()[start:query_end] > query_end
    terminal_ids = np.zeros(window.occurrence_count, dtype=np.bool_)
    terminal_ids[kept[kept >= 0]] = True
    persistent = int(retained_birth.sum()) + int(
        terminal_ids[plan.allocated_occurrences.numpy()].sum()
    )
    if query_end == n:
        return persistent + output
    recovery = req.context_recovery_plan
    cached = getattr(req, "_context_progress_demand", None)
    if cached is None or cached[0] is not program or cached[1] is not recovery:
        from .occurrence import compile_occurrence_window

        start = recovery.start if recovery is not None else len(req.prefix_indices)
        pending = np.zeros(n, dtype=np.bool_)
        for a, b in recovery.intervals if recovery is not None else ((start, n),):
            pending[a:b] = True
        canonical = program.layout.birth_positions.numpy().copy()
        source = getattr(req, "context_recovery_source", None)
        positions = source[1] if source is not None else req.context_source_positions
        if positions is not None:
            reused = ~pending[: len(positions)]
            canonical[: len(positions)][reused] = positions.numpy()[reused]
        window = compile_occurrence_window(
            program.layout, program.visible_until, program.layout.positions,
            query_start=start, query_end=n,
        )
        raw = window.occurrence_raw_tokens.numpy()
        pos = window.occurrence_positions.numpy()
        keys = window.segment_key_occurrences.numpy()
        offsets = window.segment_key_offsets.numpy()
        copies = np.zeros(n, dtype=np.int64)
        for a, b, x, y in zip(
            window.segment_query_starts.numpy(), window.segment_query_ends.numpy(),
            offsets[:-1], offsets[1:], strict=True,
        ):
            selected = keys[x:y]
            copies[a:b] = np.count_nonzero(pos[selected] != canonical[raw[selected]])
        terminal = program.layout.keep_mask.numpy() & (
            (program.layout.positions.numpy() != canonical)
            | (
                (np.arange(n) < (len(positions) if positions is not None else start))
                & (np.arange(n) >= req.context_exact_prefix_len)
                & ~pending
            )
        )
        last = keys[offsets[-2] : offsets[-1]]
        final_intermediate = int(np.count_nonzero(
            (pos[last] != canonical[raw[last]])
            & (pos[last] != program.layout.positions.numpy()[raw[last]])
        ))
        cached = req._context_progress_demand = (
            program, recovery, pending, copies, terminal, final_intermediate,
        )
    _, _, pending, copies, terminal, final_intermediate = cached
    future = pending.copy()
    future[:query_end] = False
    expiry = np.minimum(program.visible_until.numpy(), n)
    live = np.cumsum(future, dtype=np.int64) - np.bincount(
        expiry[future], minlength=n + 1,
    ).cumsum()[:n]
    demand = live + copies
    # Active terminal copies coexist with the last query's reads, but a
    # final-stage read at the terminal position uses that same occurrence.
    # The union is bounded by terminal copies plus nonterminal read copies.
    terminal = terminal.copy()
    terminal[:query_end] &= (
        req.context_window_plan[1].terminal_occurrences.numpy() < 0
    )
    demand[-1] = max(
        demand[-1], live[-1] + int(terminal.sum()) + final_intermediate,
    )
    return persistent + int(demand[query_end:].max()) + output


def handle_prefill_capacity_pressure(req, capacity, needed, tree_cache=None):
    """Distinguish self-pinned requests from transient pool pressure.

    Called only after the smallest permitted forward failed, without other
    reservations. Full-pool Context cache rows have one owner per resident raw
    token. Inspect CPU residency, never copy physical page IDs back from CUDA.
    A failed initial match owns no pages: its temporary lease can be released
    and the next pass can match from the root. Continuations must instead use
    the scheduler's deferred abort to drain work and release their ownership.
    """
    if req.context_prefill_started:
        state = req.context_state
        source_lease = getattr(req, "context_source_lease", None)
        if state is not None and (
            len(state.slots) + needed >= capacity or source_lease is not None
        ):
            # Rows are CPU ownership IDs, not GPU page numbers. Count aliases
            # once and exclude holes in a borrowed recovery gap. This is a
            # lower bound on this request's leases; other requests cannot make
            # these live canonical/terminal owners evictable.
            rows = torch.cat((state.canonical_rows, state.terminal_rows))
            pinned = len(rows[rows >= 0].unique())
            if source_lease is not None and tree_cache is not None:
                # A Retry source can keep pages outside the current state alive.
                # Count branch identities, not CUDA slot values, and only here
                # after the minimum forward has failed admission.
                pinned = max(
                    pinned,
                    tree_cache.context_leased_page_count(req)
                    + int(state.owned.count_nonzero()),
                )
            if pinned + needed >= capacity:
                req.context_admission_error = (
                    f"Context prefill retains {pinned} KV tokens and needs "
                    f"{needed} more reserved tokens, but the KV pool holds {capacity}"
                )
        return
    source = req.context_recovery_source
    slots, resident = (
        (source[0], source[2])
        if source is not None
        else (req.prefix_indices, req.context_resident)
    )
    pinned = len(slots) if resident is None else int(resident.count_nonzero())
    # Admission uses strict '<' for its total reservation, including its guard.
    if needed < capacity - pinned:
        return
    if pinned:
        req.context_force_miss = True
    else:
        req.context_admission_error = (
            f"Context prefill needs {needed} reserved KV tokens even without "
            f"a cached prefix, but the KV pool holds {capacity}"
        )


def prefill_capacity_error(req, capacity):
    """Reject impossible query read sets after matching, before acquiring KV.

    Final active length alone misses cold queries preceding a large Drop. Hot
    requests may skip those queries, so only recovery intervals count. This is
    a lower bound, not an allocation estimate (COW copies are charged separately).
    """
    if error := getattr(req, "context_admission_error", None):
        return error
    program = req.context_recompute_program or req.context_program
    if program is None or len(program.visible_until) <= capacity:
        return None
    intervals = req.context_recovery_plan.intervals
    cached = getattr(req, "_context_prefill_capacity", None)
    if cached is None or cached[0] is not program or cached[1] != intervals:
        import numpy as np

        n = len(program.visible_until)
        # Visibility metadata guarantees expiry > birth raw index. Therefore
        # every expired key was already born by the query being counted.
        expired = np.bincount(
            np.minimum(program.visible_until.numpy(), n), minlength=n + 1
        ).cumsum()
        live = np.arange(1, n + 1) - expired[:n]
        peak = max(int(live[start:end].max()) for start, end in intervals)
        cached = req._context_prefill_capacity = (program, intervals, peak)
    if cached[2] > capacity:
        return (
            f"Context prefill needs at least {cached[2]} simultaneous KV tokens "
            f"for its uncached queries, but the KV pool holds {capacity}"
        )
    return None


def request_row(pool, index):
    rows = getattr(pool, "_context_rows", None)
    if rows and index in rows:
        return rows[index]
    return pool.req_to_token[index]


def prepare_request_row(pool, req):
    if getattr(req, "context_program", None) is None:
        return
    # Native overlap may allocate a discarded decode step past the final token.
    capacity = len(req.origin_input_ids) + req.sampling_params.max_new_tokens + 4
    if capacity <= pool.req_to_token.shape[1]:
        return
    rows = getattr(pool, "_context_rows", None)
    if rows is None:
        rows = pool._context_rows = {}
    index = req.kv.req_pool_idx
    if index in rows:
        if len(rows[index]) < capacity:
            raise RuntimeError("Context raw row cannot grow while a request is live")
        return
    row = torch.full((capacity,), -1, dtype=torch.int32, device=pool.device)
    rows[index] = row
    if getattr(pool, "_context_row_pointers", None) is None:
        table = pool.req_to_token
        pool._context_row_pointers = (
            torch.arange(table.shape[0], dtype=torch.int64, device=pool.device)
            * table.stride(0)
            * table.element_size()
            + table.data_ptr()
        )
    pool._context_row_pointers[index] = row.data_ptr()


def release_request_row(pool, index):
    rows = getattr(pool, "_context_rows", None)
    if rows and index in rows:
        del rows[index]
        pool._context_row_pointers[index] = pool.req_to_token[index].data_ptr()


def row_pointers(pool):
    return pool._context_row_pointers if getattr(pool, "_context_rows", None) else None


def write_request_slots(pool, indices, values):
    rows = getattr(pool, "_context_rows", None)
    if not rows:
        pool.req_to_token[indices] = values
        return
    row, columns = indices
    if isinstance(row, int):
        request_row(pool, row)[columns] = values
    elif pool.req_to_token.is_cuda:
        import triton

        from sglang.kernels.ops.memory.context_rows import write_row_slots

        write_row_slots[(1,)](
            pool._context_row_pointers,
            row,
            columns,
            values,
            row.numel(),
            triton.next_power_of_2(row.numel()),
        )
    else:
        for index, column, value in zip(row.tolist(), columns.tolist(), values):
            request_row(pool, index)[column] = value


def validate_positions(program, limit):
    """Match mini's occurrence admission, including intermediate RoPE positions."""
    layout = program.layout
    for positions in (
        layout.birth_positions,
        layout.transition_old_positions,
        layout.transition_new_positions,
        layout.positions,
    ):
        if len(positions) and (
            int(positions.min()) < 0 or int(positions.max()) >= limit
        ):
            raise ValueError(
                "An occurrence execution position exceeds the model/RoPE limit"
            )
    active = int(layout.keep_mask.count_nonzero())
    if not active or layout.next_position <= int(
        layout.positions[layout.keep_mask].max()
    ):
        raise ValueError("Context next position must cover a nonempty active prompt")
    if layout.next_position >= limit:
        raise ValueError("Context has no room for an output token")
    return active
