"""Read-only, bounded longest-compatible search over native Radix nodes."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any


class RetryChildren:
    """Lazily built only for parents visited by Context requests.

    Native child mutations update this index once it exists. Ordinary requests
    never build it. Index entries retain exact child keys to remove a replaced
    edge even when a split has already changed that child's key.
    """

    def __init__(self, children):
        self.buckets: dict[tuple, dict[Any, Any]] = {}
        self.signatures: dict[Any, tuple] = {}
        for exact, child in children.items():
            self.link(exact, child)

    def unlink(self, exact):
        signature = self.signatures.pop(exact, None)
        if signature is not None:
            bucket = self.buckets[signature]
            del bucket[exact]
            if not bucket:
                del self.buckets[signature]

    def link(self, exact, child):
        self.unlink(exact)
        signature = child.key.context_retry_child_key()
        self.signatures[exact] = signature
        self.buckets.setdefault(signature, {})[exact] = child

    def candidates(self, key, offset):
        return self.buckets.get(key.context_retry_child_key(offset), {}).values()


@dataclass(frozen=True)
class RetrySelection:
    node: Any
    edge_length: int
    matched_length: int
    visited_edges: int = 0


def longest_compatible_prefix(root, key, page_size, advance, *, initial=None):
    """Select without splitting or changing KV/locks; materialize only the winner.

    ``advance(node, prefix_length, state)`` returns (valid, next_state), or None
    for an unavailable path. State is immutable and branch-local (in particular
    SWA suffix coverage must never leak from one candidate into another).
    ``context_descendant_bound`` is a maintained upper bound; eviction may leave
    it conservatively high, which can cost search work but cannot prune a winner.
    """
    if page_size != 1:
        raise ValueError("Context Retry requires page_size=1")
    best = initial or RetrySelection(root, 0, 0)
    frontier = []
    visited = 0

    def enqueue(parent, cursor, state):
        if cursor >= len(key):
            return
        if parent.context_retry_index is None:
            parent.context_retry_index = RetryChildren(parent.children)
        for child in parent.context_retry_index.candidates(key, cursor):
            bound = min(
                len(key), cursor + len(child.key) + child.context_descendant_bound
            )
            if bound > best.matched_length:
                heapq.heappush(frontier, (-bound, child.id, cursor, child, state))

    enqueue(root, 0, float("inf"))
    while frontier and -frontier[0][0] > best.matched_length:
        _, _, cursor, child, state = heapq.heappop(frontier)
        visited += 1
        matched = child.key.match_at(key, cursor, page_size, context_retry=True)
        if not matched:
            continue
        validity = advance(child, matched, state)
        if validity is None:
            continue
        valid, next_state = validity
        end = cursor + matched
        if valid and end > best.matched_length:
            best = RetrySelection(child, matched, end)
        if matched == len(child.key):
            enqueue(child, end, next_state)
    return RetrySelection(best.node, best.edge_length, best.matched_length, visited)
