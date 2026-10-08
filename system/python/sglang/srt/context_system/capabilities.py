"""Admission limits for the Context consumers implemented by this engine.

Run only for explicit Context requests. Native requests do not pay validation
or acquire restrictions from these staged integration limits.
"""

from __future__ import annotations


def validate_context_config(args, model_config):
    if args.page_size != 1:
        raise ValueError("Context Drop/Reposition requires page_size=1")
    architectures = model_config.hf_config.architectures or ()
    architecture = getattr(model_config, "_resolved_model_arch", None)
    if not isinstance(architecture, str):
        architecture = architectures[0] if architectures else None
    if architecture not in {
        "QWenLMHeadModel",
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "GptOssForCausalLM",
        "MiniMaxM2ForCausalLM",
    }:
        raise ValueError(f"Context Drop/Reposition is not supported by {architecture}")
    if getattr(model_config.hf_config, "dual_chunk_attention_config", None):
        raise ValueError("Context Drop/Reposition does not support DualChunk RoPE")
    if architecture == "MiniMaxM2ForCausalLM":
        config = model_config.hf_config
        head_dim = getattr(config, "head_dim", None)
        rotary_dim = getattr(config, "rotary_dim", None)
        attention_types = getattr(config, "attn_type_list", None)
        if (
            type(head_dim) is not int
            or type(rotary_dim) is not int
            or not 0 < rotary_dim <= head_dim
            or rotary_dim % 2
            or not attention_types
            or len(attention_types) != config.num_hidden_layers
            or any(kind != 1 for kind in attention_types)
        ):
            raise ValueError("Context MiniMax requires full attention and even partial RoPE")
    if model_config.is_multimodal or str(model_config.dtype) not in (
        "torch.float16",
        "torch.bfloat16",
    ):
        raise ValueError("Context requires text-only FP16/BF16 model execution")
    prefill = args.prefill_attention_backend or args.attention_backend
    decode = args.decode_attention_backend or args.attention_backend
    # FA4 shares the adapter but needs validation on supported hardware first.
    supported = {"triton", "flashinfer", "fa3"}
    if prefill not in supported or decode not in supported:
        raise ValueError("Context requires a validated backend: Triton, FlashInfer or FA3")
    if args.kv_cache_dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("Context requires unquantized FP16/BF16 KV")
    if args.disaggregation_mode != "null":
        from sglang.srt.environ import envs

        if args.disaggregation_transfer_backend != "mooncake":
            raise ValueError("Context PD currently requires Mooncake transfer")
        for name in (
            "disaggregation_decode_enable_offload_kvcache",
            "disaggregation_enable_kv_checksum",
        ):
            if getattr(args, name, False):
                raise ValueError(f"Context PD is not yet supported with {name}")
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
            raise ValueError("Context PD staging transfer is not yet supported")
    if args.pp_size != 1 or args.attn_cp_size != 1 or args.dcp_size != 1:
        raise ValueError("Context currently supports TP without PP/CP/DCP")
    if args.speculative_algorithm or args.dllm_algorithm:
        raise ValueError("Context requires autoregressive non-speculative scheduling")
    if args.enable_deterministic_inference:
        raise ValueError("Context batch-invariant attention is not yet supported")
    for name in (
        "enable_hierarchical_cache",
        "enable_unified_cache_external_linker",
        "enable_hisparse",
        "enable_lmcache",
        "enable_flexkv",
        "enable_session_radix_cache",
        "enable_beam_search",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Context is not yet supported with {name}")
    if args.radix_cache_backend is not None:
        raise ValueError("Context requires the native unified Radix cache")


def validate_context_request(args, model_config, request):
    validate_context_config(args, model_config)
    if request.input_ids is None or request.input_embeds is not None:
        raise ValueError("A Context program requires its original input_ids")
    if request.contains_mm_input() or request.session_id or request.session_params:
        raise ValueError("Context requires text input without session KV reuse")
    if request.lora_path is not None:
        raise ValueError("Context LoRA cache compatibility is not yet supported")
    if (request.sampling_params or {}).get("beam_width", 1) > 1:
        raise ValueError("Context beam scheduling is not yet supported")
    # Validate tensor IPC or native PD rebootstrap JSON before scheduler IPC.
    from sglang.srt.context_system.planner import ContextProgram

    return ContextProgram.from_wire(request.context_program, request.input_ids)
