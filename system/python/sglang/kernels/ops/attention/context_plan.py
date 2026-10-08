# MIT License
#
# Copyright (c) 2026 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU event compiler, adapted from mini-sglang's final-position compiler.

Only the compact event program is compiled here. No CUDA context, model, or
scheduler is initialized. The two native passes allocate exact transition
storage; work is linear in tokens, dropped intervals and emitted transitions.
Load this module during server startup, before accepting feature requests.
"""

from functools import cache
from hashlib import sha256


@cache
def load_context_plan():
    from tvm_ffi.cpp import load_inline

    # Include the source digest in the name so different checked-out versions
    # never accidentally share a compiled program in the same cache directory.
    return load_inline(
        "sglang_context_plan_" + sha256(_SOURCE.encode()).hexdigest()[:16],
        cpp_sources=_SOURCE,
        extra_cflags=["-std=c++20", "-O3"],
    )


_SOURCE = r"""
#include <stdexcept>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

inline void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

constexpr int32_t kToken = 0;
constexpr int32_t kDelta = 1;
constexpr int32_t kReposition = 2;

auto is_cpu_tensor(const tvm::ffi::TensorView tensor, int ndim, int bits,
                   int code = kDLInt) -> bool {
  return tensor.ndim() == ndim && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == code && tensor.dtype().bits == bits;
}

auto count_radix_reposition_transitions(
    int64_t token_count, const tvm::ffi::TensorView drop_insert_offsets,
    const tvm::ffi::TensorView drop_range_offsets,
    const tvm::ffi::TensorView drop_ranges,
    const tvm::ffi::TensorView reposition_raw_boundaries,
    const tvm::ffi::TensorView reposition_insert_offsets,
    const tvm::ffi::TensorView transition_counts,
    const tvm::ffi::TensorView status) -> void {
  require(token_count >= 0 &&
                         token_count <= std::numeric_limits<int32_t>::max(),
                     "token_count exceeds the int32 Reposition limit");
  require(is_cpu_tensor(drop_insert_offsets, 1, 32) &&
                         is_cpu_tensor(drop_range_offsets, 1, 32) &&
                         is_cpu_tensor(drop_ranges, 1, 32) &&
                         is_cpu_tensor(reposition_raw_boundaries, 1, 32) &&
                         is_cpu_tensor(reposition_insert_offsets, 1, 32) &&
                         is_cpu_tensor(transition_counts, 1, 32) &&
                         is_cpu_tensor(status, 1, 64),
                     "Reposition transition-count tensors must be contiguous CPU vectors");
  const int64_t drop_count = drop_insert_offsets.size(0);
  const int64_t reposition_count = reposition_raw_boundaries.size(0);
  require(drop_range_offsets.size(0) == drop_count + 1 &&
                         drop_ranges.size(0) % 2 == 0 &&
                         reposition_insert_offsets.size(0) == reposition_count &&
                         transition_counts.size(0) == reposition_count && status.size(0) >= 2,
                     "Reposition transition-count tensor lengths are inconsistent");

  const auto *drops = static_cast<const int32_t *>(drop_insert_offsets.data_ptr());
  const auto *drop_offsets = static_cast<const int32_t *>(drop_range_offsets.data_ptr());
  const auto *ranges = static_cast<const int32_t *>(drop_ranges.data_ptr());
  const auto *raw_boundaries =
      static_cast<const int32_t *>(reposition_raw_boundaries.data_ptr());
  const auto *reposition_offsets =
      static_cast<const int32_t *>(reposition_insert_offsets.data_ptr());
  auto *counts = static_cast<int32_t *>(transition_counts.data_ptr());
  auto *result_status = static_cast<int64_t *>(status.data_ptr());
  std::fill(counts, counts + reposition_count, 0);
  result_status[0] = 0;
  result_status[1] = -1;

  const int64_t range_count = drop_ranges.size(0) / 2;
  require(drop_offsets[0] == 0 && drop_offsets[drop_count] == range_count,
                     "drop_range_offsets does not cover drop_ranges");
  for (int64_t i = 0; i < drop_count; ++i) {
    require(drops[i] >= 0 && drops[i] <= token_count &&
                           (i == 0 || drops[i - 1] < drops[i]) &&
                           drop_offsets[i] < drop_offsets[i + 1],
                       "Invalid Drop event metadata");
  }
  for (int64_t i = 0; i < reposition_count; ++i) {
    require(reposition_offsets[i] == raw_boundaries[i] + 1 &&
                           reposition_offsets[i] > 0 &&
                           reposition_offsets[i] <= token_count &&
                           (i == 0 || raw_boundaries[i - 1] < raw_boundaries[i]),
                       "Invalid Reposition event metadata");
  }

  std::vector<int32_t> previous(token_count, -1);
  std::vector<int32_t> next(token_count, -1);
  std::vector<int32_t> positions(token_count, -1);
  std::vector<uint8_t> kept(token_count, 0);
  int32_t tail = -1;
  int32_t first_noncompact = -1;
  int32_t active_count = 0;
  int32_t next_position = 0;
  int64_t drop_idx = 0;
  int64_t reposition_idx = 0;
  for (int32_t insertion = 0; insertion <= token_count; ++insertion) {
    if (drop_idx < drop_count && drops[drop_idx] == insertion) {
      int32_t previous_end = -1;
      for (int32_t range_idx = drop_offsets[drop_idx];
           range_idx < drop_offsets[drop_idx + 1]; ++range_idx) {
        const int32_t start = ranges[2 * range_idx];
        const int32_t end = ranges[2 * range_idx + 1];
        require(start >= 0 && start < end && end <= insertion &&
                               (range_idx == drop_offsets[drop_idx] || previous_end < start),
                           "Drop ranges must be ordered, disjoint, and already materialized");
        previous_end = end;
        for (int32_t token = start; token < end; ++token) {
          if (!kept[token]) continue;
          const int32_t before = previous[token];
          const int32_t after = next[token];
          if (before >= 0) next[before] = after;
          if (after < 0) tail = before;
          else previous[after] = before;
          kept[token] = 0;
          --active_count;
          if (first_noncompact == token) first_noncompact = after;
          if (after >= 0 && (first_noncompact < 0 || after < first_noncompact))
            first_noncompact = after;
        }
      }
      if (active_count == 0) first_noncompact = -1;
      ++drop_idx;
    }
    if (reposition_idx < reposition_count &&
        reposition_offsets[reposition_idx] == insertion) {
      if (active_count == 0) {
        result_status[0] = 1;
        result_status[1] = raw_boundaries[reposition_idx];
        return;
      }
      if (first_noncompact >= 0) {
        int32_t rank = previous[first_noncompact] < 0
                           ? 0
                           : positions[previous[first_noncompact]] + 1;
        for (int32_t token = first_noncompact; token >= 0; token = next[token]) {
          require(positions[token] > rank,
                             "Non-compact active suffix invariant was violated");
          positions[token] = rank++;
          ++counts[reposition_idx];
        }
        next_position = active_count;
        first_noncompact = -1;
      }
      ++reposition_idx;
    }
    if (insertion == token_count) continue;
    const int32_t token = insertion;
    const int32_t rank = active_count;
    kept[token] = 1;
    positions[token] = next_position++;
    previous[token] = tail;
    if (tail >= 0) next[tail] = token;
    tail = token;
    ++active_count;
    if (first_noncompact < 0 && positions[token] != rank) first_noncompact = token;
  }
}

auto compile_radix_reposition_layout(
    const tvm::ffi::TensorView token_ids,
    const tvm::ffi::TensorView drop_insert_offsets,
    const tvm::ffi::TensorView drop_range_offsets,
    const tvm::ffi::TensorView drop_ranges,
    const tvm::ffi::TensorView reposition_raw_boundaries,
    const tvm::ffi::TensorView reposition_insert_offsets,
    const tvm::ffi::TensorView records,
    const tvm::ffi::TensorView virtual_mask,
    const tvm::ffi::TensorView key_to_token,
    const tvm::ffi::TensorView token_to_key,
    const tvm::ffi::TensorView positions,
    const tvm::ffi::TensorView keep_mask,
    const tvm::ffi::TensorView materialized_stage,
    const tvm::ffi::TensorView birth_positions,
    const tvm::ffi::TensorView birth_stages,
    const tvm::ffi::TensorView transition_offsets,
    const tvm::ffi::TensorView transition_raw_tokens,
    const tvm::ffi::TensorView transition_old_positions,
    const tvm::ffi::TensorView transition_new_positions,
    const tvm::ffi::TensorView effective_reposition_stages,
    const tvm::ffi::TensorView drop_event_to_key,
    const tvm::ffi::TensorView effective_repositions,
    const tvm::ffi::TensorView ignored_repositions,
    const tvm::ffi::TensorView status) -> void {
  require(is_cpu_tensor(token_ids, 1, 32) ||
                         is_cpu_tensor(token_ids, 1, 64),
                     "token_ids must be a CPU int32/int64 vector");
  require(is_cpu_tensor(drop_insert_offsets, 1, 32) &&
                         is_cpu_tensor(drop_range_offsets, 1, 32) &&
                         is_cpu_tensor(drop_ranges, 1, 32),
                     "Drop inputs must be contiguous CPU int32 vectors");
  require(is_cpu_tensor(reposition_raw_boundaries, 1, 32) &&
                         is_cpu_tensor(reposition_insert_offsets, 1, 32),
                     "Reposition inputs must be contiguous CPU int32 vectors");
  require(is_cpu_tensor(records, 2, 32) && records.size(1) == 3,
                     "records must be a contiguous CPU int32 [N, 3] tensor");
  require(is_cpu_tensor(virtual_mask, 1, 8, kDLBool) &&
                         is_cpu_tensor(keep_mask, 1, 8, kDLBool) &&
                         is_cpu_tensor(effective_repositions, 1, 8, kDLBool) &&
                         is_cpu_tensor(ignored_repositions, 1, 8, kDLBool),
                     "mask outputs must be contiguous CPU bool vectors");
  require(is_cpu_tensor(key_to_token, 1, 64) &&
                         is_cpu_tensor(token_to_key, 1, 64) &&
                         is_cpu_tensor(drop_event_to_key, 1, 64) &&
                         is_cpu_tensor(status, 1, 64),
                     "mapping/status outputs must be contiguous CPU int64 vectors");
  require(is_cpu_tensor(positions, 1, 32) &&
                         is_cpu_tensor(materialized_stage, 1, 32) &&
                         is_cpu_tensor(birth_positions, 1, 32) &&
                         is_cpu_tensor(birth_stages, 1, 32) &&
                         is_cpu_tensor(transition_offsets, 1, 32) &&
                         is_cpu_tensor(transition_raw_tokens, 1, 32) &&
                         is_cpu_tensor(transition_old_positions, 1, 32) &&
                         is_cpu_tensor(transition_new_positions, 1, 32) &&
                         is_cpu_tensor(effective_reposition_stages, 1, 32),
                     "token metadata outputs must be contiguous CPU int32 vectors");

  const int64_t token_count = token_ids.size(0);
  const int64_t drop_count = drop_insert_offsets.size(0);
  const int64_t reposition_count = reposition_raw_boundaries.size(0);
  require(token_count <= std::numeric_limits<int32_t>::max(),
                     "The token stream exceeds the int32 Reposition limit");
  require(drop_range_offsets.size(0) == drop_count + 1,
                     "drop_range_offsets must have E + 1 entries");
  require(drop_ranges.size(0) % 2 == 0,
                     "drop_ranges must contain start/end pairs");
  require(reposition_insert_offsets.size(0) == reposition_count,
                     "Reposition boundaries and offsets must align");
  require(records.size(0) >=
                         token_count + drop_ranges.size(0) / 2 + reposition_count &&
                         virtual_mask.size(0) >= records.size(0) &&
                         key_to_token.size(0) >= records.size(0),
                     "Radix output capacity is too small");
  require(token_to_key.size(0) == token_count &&
                         positions.size(0) == token_count &&
                         keep_mask.size(0) == token_count &&
                         materialized_stage.size(0) == token_count &&
                         birth_positions.size(0) == token_count &&
                         birth_stages.size(0) == token_count,
                     "Per-token output lengths are inconsistent");
  require(effective_repositions.size(0) == reposition_count &&
                         ignored_repositions.size(0) == reposition_count &&
                         effective_reposition_stages.size(0) == reposition_count &&
                         drop_event_to_key.size(0) == drop_count &&
                         transition_offsets.size(0) >= reposition_count + 1 &&
                         transition_raw_tokens.size(0) == transition_old_positions.size(0) &&
                         transition_raw_tokens.size(0) == transition_new_positions.size(0) &&
                         status.size(0) >= 5,
                     "Reposition output lengths are inconsistent");

  const auto *drops = static_cast<const int32_t *>(drop_insert_offsets.data_ptr());
  const auto *drop_offsets = static_cast<const int32_t *>(drop_range_offsets.data_ptr());
  const auto *ranges = static_cast<const int32_t *>(drop_ranges.data_ptr());
  const auto *raw_boundaries =
      static_cast<const int32_t *>(reposition_raw_boundaries.data_ptr());
  const auto *reposition_offsets =
      static_cast<const int32_t *>(reposition_insert_offsets.data_ptr());
  const int64_t range_count = drop_ranges.size(0) / 2;

  require(drop_offsets[0] == 0 && drop_offsets[drop_count] == range_count,
                     "drop_range_offsets does not cover drop_ranges");
  for (int64_t i = 0; i < drop_count; ++i) {
    require(drops[i] >= 0 && drops[i] <= token_count,
                       "Drop insertion offset is outside the token stream");
    require(i == 0 || drops[i - 1] < drops[i],
                       "Drop insertion offsets must be strictly increasing");
    require(drop_offsets[i] < drop_offsets[i + 1],
                       "Every Drop event must own a non-empty range set");
  }
  for (int64_t i = 0; i < reposition_count; ++i) {
    require(reposition_offsets[i] == raw_boundaries[i] + 1 &&
                           reposition_offsets[i] > 0 &&
                           reposition_offsets[i] <= token_count,
                       "Reposition raw boundary and insertion offset disagree");
    require(i == 0 || raw_boundaries[i - 1] < raw_boundaries[i],
                       "Reposition raw boundaries must be strictly increasing");
  }

  auto *position = static_cast<int32_t *>(positions.data_ptr());
  auto *kept = static_cast<bool *>(keep_mask.data_ptr());
  auto *ready = static_cast<int32_t *>(materialized_stage.data_ptr());
  auto *birth_position = static_cast<int32_t *>(birth_positions.data_ptr());
  auto *birth_stage = static_cast<int32_t *>(birth_stages.data_ptr());
  auto *transition_offset = static_cast<int32_t *>(transition_offsets.data_ptr());
  auto *transition_token = static_cast<int32_t *>(transition_raw_tokens.data_ptr());
  auto *transition_old = static_cast<int32_t *>(transition_old_positions.data_ptr());
  auto *transition_new = static_cast<int32_t *>(transition_new_positions.data_ptr());
  auto *reposition_stage =
      static_cast<int32_t *>(effective_reposition_stages.data_ptr());
  auto *drop_key = static_cast<int64_t *>(drop_event_to_key.data_ptr());
  auto *effective = static_cast<bool *>(effective_repositions.data_ptr());
  auto *ignored = static_cast<bool *>(ignored_repositions.data_ptr());
  auto *result_status = static_cast<int64_t *>(status.data_ptr());
  std::fill(effective, effective + reposition_count, false);
  std::fill(ignored, ignored + reposition_count, false);
  std::fill(reposition_stage, reposition_stage + reposition_count, -1);
  transition_offset[0] = 0;
  int32_t transition_cursor = 0;

  std::vector<int32_t> previous(token_count, -1);
  std::vector<int32_t> next(token_count, -1);
  int32_t head = -1;
  int32_t tail = -1;
  int32_t first_noncompact = -1;
  int32_t active_count = 0;
  int32_t next_position = 0;
  int32_t stage = 0;
  int64_t drop_idx = 0;
  int64_t reposition_idx = 0;

  for (int32_t insertion = 0; insertion <= token_count; ++insertion) {
    if (drop_idx < drop_count && drops[drop_idx] == insertion) {
      int32_t previous_end = -1;
      for (int32_t range_idx = drop_offsets[drop_idx];
           range_idx < drop_offsets[drop_idx + 1]; ++range_idx) {
        const int32_t start = ranges[2 * range_idx];
        const int32_t end = ranges[2 * range_idx + 1];
        require(start >= 0 && start < end && end <= insertion &&
                               (range_idx == drop_offsets[drop_idx] || previous_end < start),
                           "Drop ranges must be ordered, disjoint, and already materialized");
        previous_end = end;
        for (int32_t token = start; token < end; ++token) {
          if (!kept[token]) {
            continue;
          }
          const int32_t before = previous[token];
          const int32_t after = next[token];
          if (before < 0) {
            head = after;
          } else {
            next[before] = after;
          }
          if (after < 0) {
            tail = before;
          } else {
            previous[after] = before;
          }
          kept[token] = false;
          --active_count;
          if (first_noncompact == token) {
            first_noncompact = after;
          }
          if (after >= 0 &&
              (first_noncompact < 0 || after < first_noncompact)) {
            first_noncompact = after;
          }
        }
      }
      if (active_count == 0) {
        first_noncompact = -1;
      }
      ++drop_idx;
    }

    if (reposition_idx < reposition_count &&
        reposition_offsets[reposition_idx] == insertion) {
      if (active_count == 0) {
        result_status[0] = 1;
        result_status[4] = raw_boundaries[reposition_idx];
        return;
      }
      if (first_noncompact < 0) {
        ignored[reposition_idx] = true;
      } else {
        ++stage;
        effective[reposition_idx] = true;
        reposition_stage[reposition_idx] = stage;
        int32_t rank = previous[first_noncompact] < 0
                           ? 0
                           : position[previous[first_noncompact]] + 1;
        for (int32_t token = first_noncompact; token >= 0; token = next[token]) {
          require(position[token] > rank,
                             "Non-compact active suffix invariant was violated");
          const int32_t old_position = position[token];
          const int32_t new_position = rank++;
          require(transition_cursor < transition_raw_tokens.size(0),
                             "Reposition transition output capacity is too small");
          transition_token[transition_cursor] = token;
          transition_old[transition_cursor] = old_position;
          transition_new[transition_cursor] = new_position;
          ++transition_cursor;
          position[token] = new_position;
          ready[token] = stage;
        }
        next_position = active_count;
        first_noncompact = -1;
        transition_offset[stage] = transition_cursor;
      }
      ++reposition_idx;
    }

    if (insertion == token_count) {
      continue;
    }
    const int32_t token = insertion;
    const int32_t rank = active_count;
    kept[token] = true;
    position[token] = next_position++;
    birth_position[token] = position[token];
    birth_stage[token] = stage;
    ready[token] = stage;
    previous[token] = tail;
    if (tail < 0) {
      head = token;
    } else {
      next[tail] = token;
    }
    tail = token;
    ++active_count;
    if (first_noncompact < 0 && position[token] != rank) {
      first_noncompact = token;
    }
  }

  auto *output = static_cast<int32_t *>(records.data_ptr());
  auto *is_virtual = static_cast<bool *>(virtual_mask.data_ptr());
  auto *key_token = static_cast<int64_t *>(key_to_token.data_ptr());
  auto *token_key = static_cast<int64_t *>(token_to_key.data_ptr());
  const auto *tokens32 = token_ids.dtype().bits == 32
                             ? static_cast<const int32_t *>(token_ids.data_ptr())
                             : nullptr;
  const auto *tokens64 = token_ids.dtype().bits == 64
                             ? static_cast<const int64_t *>(token_ids.data_ptr())
                             : nullptr;
  int64_t key_idx = 0;
  drop_idx = 0;
  reposition_idx = 0;
  for (int32_t insertion = 0; insertion <= token_count; ++insertion) {
    if (drop_idx < drop_count && drops[drop_idx] == insertion) {
      drop_key[drop_idx] = key_idx;
      for (int32_t range_idx = drop_offsets[drop_idx];
           range_idx < drop_offsets[drop_idx + 1]; ++range_idx) {
        const int32_t start = ranges[2 * range_idx];
        const int32_t end = ranges[2 * range_idx + 1];
        output[key_idx * 3] = kDelta;
        output[key_idx * 3 + 1] =
            static_cast<int32_t>(-static_cast<int64_t>(start) - 1);
        output[key_idx * 3 + 2] =
            static_cast<int32_t>(-static_cast<int64_t>(end) - 1);
        is_virtual[key_idx] = true;
        key_token[key_idx++] = -1;
      }
      ++drop_idx;
    }
    if (reposition_idx < reposition_count &&
        reposition_offsets[reposition_idx] == insertion) {
      if (effective[reposition_idx]) {
        output[key_idx * 3] = kReposition;
        output[key_idx * 3 + 1] = raw_boundaries[reposition_idx];
        output[key_idx * 3 + 2] = -1;
        is_virtual[key_idx] = true;
        key_token[key_idx++] = -1;
      }
      ++reposition_idx;
    }
    if (insertion == token_count) {
      continue;
    }
    token_key[insertion] = key_idx;
    const int64_t token_id = tokens32 == nullptr ? tokens64[insertion]
                                                  : tokens32[insertion];
    if (token_id < 0 || token_id > std::numeric_limits<int32_t>::max()) {
      result_status[0] = 2;
      result_status[4] = token_id;
      return;
    }
    output[key_idx * 3] = kToken;
    output[key_idx * 3 + 1] = static_cast<int32_t>(token_id);
    output[key_idx * 3 + 2] = position[insertion];
    is_virtual[key_idx] = false;
    key_token[key_idx++] = insertion;
  }

  result_status[0] = 0;
  result_status[1] = key_idx;
  result_status[2] = stage;
  result_status[3] = next_position;
  result_status[4] = -1;
  require(transition_cursor == transition_raw_tokens.size(0),
                     "Reposition transition count/fill passes disagree");
  (void)head;
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(compile_radix_reposition_layout,
                              compile_radix_reposition_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(count_radix_reposition_transitions,
                              count_radix_reposition_transitions);
"""


_TEXT_MATCH_SOURCE = r"""
#include <stdexcept>

#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

inline void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}


auto is_cpu_int_tensor(const tvm::ffi::TensorView tensor, int ndim,
                       int bits) -> bool {
  return tensor.ndim() == ndim && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == kDLInt && tensor.dtype().bits == bits;
}

struct Node {
  std::unordered_map<int32_t, int32_t> next;
  int32_t failure = 0;
  int32_t output_link = -1;
  std::vector<int32_t> terminal_patterns;
};

auto transition(const std::vector<Node> &trie, int32_t node,
                int32_t byte) -> int32_t {
  const auto found = trie[node].next.find(byte);
  return found == trie[node].next.end() ? -1 : found->second;
}

auto aho_find_all(const tvm::ffi::TensorView source,
                  const tvm::ffi::TensorView flat_patterns,
                  const tvm::ffi::TensorView pattern_offsets,
                  const tvm::ffi::TensorView matches,
                  int64_t max_matches) -> int64_t {
  require(is_cpu_int_tensor(source, 1, 32),
                     "source must be a contiguous 1D CPU int32 tensor");
  require(is_cpu_int_tensor(flat_patterns, 1, 32),
                     "flat_patterns must be a contiguous 1D CPU int32 tensor");
  require(is_cpu_int_tensor(pattern_offsets, 1, 64),
                     "pattern_offsets must be a contiguous 1D CPU int64 tensor");
  require(is_cpu_int_tensor(matches, 2, 64) && matches.size(1) == 3,
                     "matches must be a contiguous [N, 3] CPU int64 tensor");
  require(max_matches > 0 && matches.size(0) >= max_matches,
                     "matches capacity is smaller than max_matches");
  require(pattern_offsets.size(0) >= 1,
                     "pattern_offsets must contain at least zero");

  const auto *text = static_cast<const int32_t *>(source.data_ptr());
  const auto *patterns = static_cast<const int32_t *>(flat_patterns.data_ptr());
  const auto *offsets = static_cast<const int64_t *>(pattern_offsets.data_ptr());
  auto *result = static_cast<int64_t *>(matches.data_ptr());
  const int64_t pattern_count = pattern_offsets.size(0) - 1;

  std::vector<Node> trie(1);
  std::vector<int64_t> lengths(pattern_count);
  for (int64_t pattern_id = 0; pattern_id < pattern_count; ++pattern_id) {
    const int64_t begin = offsets[pattern_id];
    const int64_t end = offsets[pattern_id + 1];
    require(begin >= 0 && end > begin &&
                           end <= flat_patterns.size(0),
                       "pattern_offsets contains an invalid or empty pattern");
    lengths[pattern_id] = end - begin;
    int32_t node = 0;
    for (int64_t pos = begin; pos < end; ++pos) {
      const int32_t byte = patterns[pos];
      require(byte >= 0 && byte <= 255,
                         "patterns must contain byte values in [0, 255]");
      const auto found = trie[node].next.find(byte);
      if (found == trie[node].next.end()) {
        const auto child = static_cast<int32_t>(trie.size());
        trie[node].next.emplace(byte, child);
        trie.emplace_back();
        node = child;
      } else {
        node = found->second;
      }
    }
    trie[node].terminal_patterns.push_back(static_cast<int32_t>(pattern_id));
  }

  std::queue<int32_t> queue;
  for (const auto &[byte, child] : trie[0].next) {
    (void)byte;
    trie[child].failure = 0;
    queue.push(child);
  }
  while (!queue.empty()) {
    const int32_t node = queue.front();
    queue.pop();
    for (const auto &[byte, child] : trie[node].next) {
      int32_t fallback = trie[node].failure;
      int32_t fallback_child = transition(trie, fallback, byte);
      while (fallback != 0 && fallback_child == -1) {
        fallback = trie[fallback].failure;
        fallback_child = transition(trie, fallback, byte);
      }
      if (fallback_child != -1) {
        fallback = fallback_child;
      }
      trie[child].failure = fallback;
      trie[child].output_link = !trie[fallback].terminal_patterns.empty()
                                    ? fallback
                                    : trie[fallback].output_link;
      queue.push(child);
    }
  }

  int64_t count = 0;
  int32_t node = 0;
  for (int64_t pos = 0; pos < source.size(0); ++pos) {
    const int32_t byte = text[pos];
    require(byte >= 0 && byte <= 255,
                       "source must contain byte values in [0, 255]");
    int32_t child = transition(trie, node, byte);
    while (node != 0 && child == -1) {
      node = trie[node].failure;
      child = transition(trie, node, byte);
    }
    if (child != -1) {
      node = child;
    }
    for (int32_t output_node = node; output_node != -1;
         output_node = trie[output_node].output_link) {
      for (const int32_t pattern_id : trie[output_node].terminal_patterns) {
        if (count >= max_matches) {
          return -1;
        }
        result[count * 3] = pattern_id;
        result[count * 3 + 1] = pos + 1 - lengths[pattern_id];
        result[count * 3 + 2] = pos + 1;
        ++count;
      }
    }
  }
  return count;
}

auto ordered_latest_find(const tvm::ffi::TensorView flat_sources,
                         const tvm::ffi::TensorView source_offsets,
                         const tvm::ffi::TensorView source_keys,
                         const tvm::ffi::TensorView flat_patterns,
                         const tvm::ffi::TensorView pattern_offsets,
                         const tvm::ffi::TensorView pattern_keys,
                         const tvm::ffi::TensorView matches) -> int64_t {
  require(is_cpu_int_tensor(flat_sources, 1, 32),
                     "flat_sources must be a contiguous 1D CPU int32 tensor");
  require(is_cpu_int_tensor(source_offsets, 1, 64),
                     "source_offsets must be a contiguous 1D CPU int64 tensor");
  require(is_cpu_int_tensor(source_keys, 1, 64),
                     "source_keys must be a contiguous 1D CPU int64 tensor");
  require(is_cpu_int_tensor(flat_patterns, 1, 32),
                     "flat_patterns must be a contiguous 1D CPU int32 tensor");
  require(is_cpu_int_tensor(pattern_offsets, 1, 64),
                     "pattern_offsets must be a contiguous 1D CPU int64 tensor");
  require(is_cpu_int_tensor(pattern_keys, 1, 64),
                     "pattern_keys must be a contiguous 1D CPU int64 tensor");
  require(is_cpu_int_tensor(matches, 2, 64) && matches.size(1) == 3,
                     "matches must be a contiguous [N, 3] CPU int64 tensor");
  require(source_offsets.size(0) == source_keys.size(0) + 1,
                     "source offsets and keys have inconsistent lengths");
  require(pattern_offsets.size(0) == pattern_keys.size(0) + 1,
                     "pattern offsets and keys have inconsistent lengths");
  require(matches.size(0) == pattern_keys.size(0),
                     "matches must provide one row per pattern");

  const auto *sources = static_cast<const int32_t *>(flat_sources.data_ptr());
  const auto *source_starts =
      static_cast<const int64_t *>(source_offsets.data_ptr());
  const auto *source_key_values =
      static_cast<const int64_t *>(source_keys.data_ptr());
  const auto *patterns = static_cast<const int32_t *>(flat_patterns.data_ptr());
  const auto *pattern_starts =
      static_cast<const int64_t *>(pattern_offsets.data_ptr());
  const auto *pattern_key_values =
      static_cast<const int64_t *>(pattern_keys.data_ptr());
  auto *result = static_cast<int64_t *>(matches.data_ptr());

  int64_t source_id = source_keys.size(0) - 1;
  for (int64_t pattern_id = pattern_keys.size(0) - 1; pattern_id >= 0;
       --pattern_id) {
    const int64_t pattern_begin = pattern_starts[pattern_id];
    const int64_t pattern_end = pattern_starts[pattern_id + 1];
    const int64_t pattern_size = pattern_end - pattern_begin;
    require(pattern_begin >= 0 && pattern_end >= pattern_begin &&
                           pattern_end <= flat_patterns.size(0),
                       "pattern_offsets contains an invalid range");

    std::vector<int64_t> prefix(pattern_size, 0);
    for (int64_t i = 1, matched = 0; i < pattern_size; ++i) {
      const int32_t value = patterns[pattern_end - 1 - i];
      while (matched > 0 &&
             value != patterns[pattern_end - 1 - matched]) {
        matched = prefix[matched - 1];
      }
      if (value == patterns[pattern_end - 1 - matched]) {
        ++matched;
      }
      prefix[i] = matched;
    }

    bool found = false;
    while (source_id >= 0 && !found) {
      const int64_t current_source = source_id--;
      if (source_key_values[current_source] != pattern_key_values[pattern_id]) {
        continue;
      }
      const int64_t source_begin = source_starts[current_source];
      const int64_t source_end = source_starts[current_source + 1];
      require(source_begin >= 0 && source_end >= source_begin &&
                             source_end <= flat_sources.size(0),
                         "source_offsets contains an invalid range");

      if (pattern_size == 0) {
        result[pattern_id * 3] = current_source;
        result[pattern_id * 3 + 1] = source_end - source_begin;
        result[pattern_id * 3 + 2] = source_end - source_begin;
        found = true;
        break;
      }

      int64_t matched = 0;
      for (int64_t pos = source_end; pos > source_begin; --pos) {
        const int32_t value = sources[pos - 1];
        while (matched > 0 &&
               value != patterns[pattern_end - 1 - matched]) {
          matched = prefix[matched - 1];
        }
        if (value == patterns[pattern_end - 1 - matched]) {
          ++matched;
        }
        if (matched == pattern_size) {
          const int64_t local_start = pos - source_begin - 1;
          result[pattern_id * 3] = current_source;
          result[pattern_id * 3 + 1] = local_start;
          result[pattern_id * 3 + 2] = local_start + pattern_size;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      return pattern_id;
    }
  }
  return -1;
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(aho_find_all, aho_find_all);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ordered_latest_find, ordered_latest_find);
"""


@cache
def load_context_text_match():
    """Load the linear UTF-8 selector compiler before accepting Context work."""
    from tvm_ffi.cpp import load_inline

    digest = sha256(_TEXT_MATCH_SOURCE.encode()).hexdigest()[:16]
    return load_inline(
        name=f"sglang_context_text_match_{digest}",
        cpp_sources=_TEXT_MATCH_SOURCE,
        extra_cflags=["-std=c++20", "-O3"],
    )
