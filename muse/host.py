"""Prepare before task start; attach to the unchanged V5 HookClient."""

import asyncio
from pathlib import Path
from .io import configuration, verify_manifest, PACKAGE_ROOT
from .tldr import prepare_tldr
from .runtime import V8Runtime
from .native import make_step_callback
from .bank import verify_v8_bank


async def prepare_callback(
    bank,
    out,
    original_instruction,
    tldr_cache,
    gate_llm,
    tldr_llm,
    conf=None,
    renderer=None,
    assistant_provider=None,
    record_transform=None,
    initial_source_refs=None,
    route_bank=None,
):
    if (PACKAGE_ROOT / "MANIFEST.json").exists():
        verify_manifest(PACKAGE_ROOT)
    verify_v8_bank(bank, require_initial=True)
    config = configuration(conf)
    if (
        gate_llm.model != config["gate_model"]
        or gate_llm.effort != config["gate_effort"]
    ):
        raise ValueError("Gate model/effort does not match V8 config")
    if (
        tldr_llm.model != config["tldr_model"]
        or tldr_llm.effort != config["tldr_effort"]
    ):
        raise ValueError("TLDR model/effort does not match V8 config")
    if config.get("gate_enabled", True):
        summary = await asyncio.to_thread(
            prepare_tldr, original_instruction, tldr_cache, tldr_llm
        )
    else:
        summary = {
            "tldr": "",
            "cache_hit": True,
            "cost_cny": 0,
            "not_required": "direct_top1_without_gate",
        }

    runtime = V8Runtime(
        bank,
        Path(out) / "v8",
        summary,
        gate_llm,
        config,
        renderer,
        initial_source_refs=initial_source_refs,
        route_bank=route_bank,
    )
    return runtime, make_step_callback(runtime, assistant_provider, record_transform)
