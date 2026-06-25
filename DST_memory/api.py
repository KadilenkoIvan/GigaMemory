"""GigaMemory FastAPI REST server.

Endpoints:
  POST   /dialogue/{dialogue_id}/message        — send message, get LLM answer
  GET    /dialogue/{dialogue_id}/graph          — full memory graph (JSON)
  GET    /dialogue/{dialogue_id}/graph_short    — compact: active triplets + TTL deadline only
  GET    /dialogue/{dialogue_id}/graph/image    — memory graph as PNG
  DELETE /dialogue/{dialogue_id}               — reset dialogue memory
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared app state
# ---------------------------------------------------------------------------


class _State:
    pipeline: Any = None
    session_dir: str = ""
    _dialogue_locks: dict[str, threading.Lock] = {}
    _locks_mutex: threading.Lock = threading.Lock()

    def lock_for(self, dialogue_id: str) -> threading.Lock:
        with self._locks_mutex:
            if dialogue_id not in self._dialogue_locks:
                self._dialogue_locks[dialogue_id] = threading.Lock()
            return self._dialogue_locks[dialogue_id]


_state = _State()

# ---------------------------------------------------------------------------
# Pipeline bootstrap
# ---------------------------------------------------------------------------


def _ensure_ragu_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    local_ragu = repo_root / "RAGU"
    if local_ragu.is_dir():
        p = str(local_ragu.resolve())
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_pipeline(config_path: str) -> tuple[Any, str]:
    """Return (pipeline, session_dir) built from *config_path*."""
    _ensure_ragu_on_path()

    from dst_memory import PipelineConfig
    from dst_memory.core.pipeline import DSTMemoryPipeline
    from dst_memory.storage.ragu_graph_processor import build_ragu_processor
    from dst_memory.utils.run_config_loader import (
        load_run_config,
        shared_section,
        subsection,
    )

    raw = load_run_config(config_path)
    s = shared_section(raw)
    api = subsection(raw, "api")

    session_dir = str(api.get("session_dir", "") or "")
    ragu_storage = str(
        api.get("ragu_storage_path", "") or s.get("ragu_storage_path", "") or ""
    )

    cfg = PipelineConfig(
        importance_model_path=str(s.get("importance_model_path", "")),
        importance_threshold=float(s.get("importance_threshold", 0.25)),
        retrieval_top_k=int(s.get("retrieval_top_k", 5)),
        graph_top_k_records=int(s.get("graph_top_k_records", 20)),
        recent_history_pairs=int(s.get("recent_history_pairs", 5)),
        use_memory_gate=not bool(s.get("disable_memory_gate", False)),
        memory_gate_use_stub=bool(s.get("memory_gate_use_stub", False)),
        memory_strategy=str(s.get("memory_strategy", "relevant_slots_full")),
        llm_mode=str(s.get("llm_mode", "openrouter")),
        llm_api_url=str(s.get("llm_api_url", "https://openrouter.ai/api/v1")),
        llm_api_key=str(s.get("llm_api_key", "")),
        llm_model=str(s.get("llm_model", "")),
        llm_tokenizer_model=str(s.get("llm_tokenizer_model", "")),
        llm_max_tokens=int(s.get("llm_max_tokens", 1024)),
        llm_temperature=float(s.get("llm_temperature", 0.0)),
        llm_enable_thinking=bool(s.get("llm_enable_thinking", False)),
        openrouter_http_referer=str(s.get("openrouter_http_referer", "")),
        openrouter_x_title=str(s.get("openrouter_x_title", "")),
        slot_use_stub=bool(s.get("slot_use_stub", False)),
        slot_model_path=str(s.get("slot_model_path", "")),
        slot_max_slots_per_message=int(s.get("slot_max_slots_per_message", 5)),
        slot_model_enable_thinking=bool(s.get("slot_model_enable_thinking", False)),
        slot_llm_inject_no_think_prompt=bool(
            s.get("slot_llm_inject_no_think_prompt", True)
        ),
        slot_llm_lm_format_enforcer=bool(s.get("slot_llm_lm_format_enforcer", False)),
        slot_llm_load_quantization=str(
            s.get("slot_llm_load_quantization", "none") or "none"
        ),
        use_ragu=True,
        ragu_embedder_model=str(s.get("ragu_embedder_model", "deepvk/USER-bge-m3")),
        ragu_storage_path=ragu_storage,
        ttl_mode=str(s.get("ttl_mode", "mode2")),
        ttl_slot_overrides=json.loads(str(s.get("ttl_slot_overrides", "{}") or "{}")),
        ttl_semantic_dedup_enabled=bool(s.get("ttl_semantic_dedup_enabled", True)),
        ttl_semantic_dedup_threshold=float(s.get("ttl_semantic_dedup_threshold", 0.9)),
        slot_context_enabled=bool(s.get("slot_context_enabled", False)),
        slot_context_max_facts=int(s.get("slot_context_max_facts", 10)),
        triplet_deletion_mode=str(s.get("triplet_deletion_mode", "none")),
        deletion_use_pymorphy=bool(s.get("deletion_use_pymorphy", False)),
        conflict_allow_multi_relation_same_object=bool(
            s.get("conflict_allow_multi_relation_same_object", True)
        ),
        conflict_rule_same_relation_updates=bool(
            s.get("conflict_rule_same_relation_updates", True)
        ),
        slot_fallback_on_no_slots=bool(s.get("slot_fallback_on_no_slots", True)),
        triplet_fallback_on_empty=bool(s.get("triplet_fallback_on_empty", True)),
        prompt_language=str(s.get("prompt_language", "ru")),
        use_dataset_datetime=bool(s.get("use_dataset_datetime", False)),
        force_infinite_ttl=bool(s.get("force_infinite_ttl", True)),
        parallel_write_mode=False,
        session_dir=session_dir,
    )

    logger.info(
        "Loading RAGU backend embedder=%s storage=%s",
        cfg.ragu_embedder_model,
        cfg.ragu_storage_path or "<in-memory>",
    )
    _kg, ragu_processor = build_ragu_processor(
        embedder_model=cfg.ragu_embedder_model,
        storage_path=cfg.ragu_storage_path or None,
    )
    pipeline = DSTMemoryPipeline(cfg, ragu_processor=ragu_processor)
    pipeline.final_llm.realtime_mode = True
    return pipeline, session_dir


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get(
        "GIGAMEMORY_CONFIG",
        str(Path(__file__).parent / "run_config_api.json"),
    )
    logger.info("GigaMemory API starting — config=%s", config_path)
    _state.pipeline, _state.session_dir = _load_pipeline(config_path)
    logger.info("Pipeline ready. session_dir=%r", _state.session_dir or "(none)")
    yield
    logger.info("GigaMemory API shutting down")


app = FastAPI(
    title="GigaMemory API",
    description=(
        "Long-term memory REST API. "
        "Wraps the GigaMemory pipeline: DST graph + RAGU retrieval + final LLM."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class MessageRequest(BaseModel):
    content: str
    parallel_write: bool = False


class MessageResponse(BaseModel):
    dialogue_id: str
    answer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_pipeline() -> Any:
    if _state.pipeline is None:
        raise HTTPException(503, detail="Pipeline not initialized")
    return _state.pipeline


def _save_session(dialogue_id: str) -> None:
    if _state.session_dir and _state.pipeline is not None:
        try:
            _state.pipeline.save_session(dialogue_id, _state.session_dir)
        except Exception as e:
            logger.warning("Session save failed dialogue_id=%s: %s", dialogue_id, e)


def _ttl_expires_at(ttl: str, created_at_datetime: str) -> str | None:
    """Return ISO expiry datetime or None for 'inf'."""
    from dst_memory.core.models import TTL_TO_TIMEDELTA

    delta = TTL_TO_TIMEDELTA.get(ttl)
    if delta is None:
        return None
    try:
        created = datetime.fromisoformat(created_at_datetime)
        return (created + delta).isoformat()
    except (ValueError, TypeError):
        return None


def _slot_color_palette(slot_names: list[str]) -> dict[str, str]:
    palette = [
        "#E74C3C",
        "#3498DB",
        "#2ECC71",
        "#F39C12",
        "#9B59B6",
        "#1ABC9C",
        "#E67E22",
        "#34495E",
        "#E91E63",
        "#00BCD4",
    ]
    return {
        name: palette[i % len(palette)] for i, name in enumerate(sorted(slot_names))
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/dialogue/{dialogue_id}/message", response_model=MessageResponse)
def post_message(dialogue_id: str, req: MessageRequest) -> MessageResponse:
    """Send a user message and receive an LLM answer.

    In **parallel_write** mode the memory graph is updated in a background
    thread while the answer is generated immediately from the current graph state.
    """
    from dst_memory.core.models import Message

    pipeline = _require_pipeline()
    msg = Message(role="user", content=req.content)

    if req.parallel_write:

        def _bg_write() -> None:
            try:
                pipeline.write_to_memory(dialogue_id, msg)
                _save_session(dialogue_id)
            except Exception as e:
                logger.error(
                    "Background write failed dialogue_id=%s: %s", dialogue_id, e
                )

        t = threading.Thread(target=_bg_write, daemon=True)
        t.start()
        answer = pipeline.answer(dialogue_id, req.content)
        pipeline.add_recent_pair(dialogue_id, req.content, answer)
    else:
        lock = _state.lock_for(dialogue_id)
        with lock:
            pipeline.write_to_memory(dialogue_id, msg)
            answer = pipeline.answer(dialogue_id, req.content)
            pipeline.add_recent_pair(dialogue_id, req.content, answer)
            _save_session(dialogue_id)

    return MessageResponse(dialogue_id=dialogue_id, answer=answer)


@app.get("/dialogue/{dialogue_id}/graph")
def get_graph(dialogue_id: str) -> JSONResponse:
    """Return the full memory graph as JSON (active facts with all metadata)."""
    pipeline = _require_pipeline()
    slots = pipeline.dst.slots_with_messages(dialogue_id)
    return JSONResponse({"dialogue_id": dialogue_id, "slots": slots})


@app.get("/dialogue/{dialogue_id}/graph_short")
def get_graph_short(dialogue_id: str) -> JSONResponse:
    """Return a compact memory graph: only active triplets per slot with TTL deadline.

    Each record contains: subject, relation, object, ttl (label), expires_at (ISO or null).
    No record IDs, source texts, step counters, or other metadata.
    """
    pipeline = _require_pipeline()
    state = pipeline.dst.get_state(dialogue_id)

    result: dict[str, list[dict]] = {}
    for slot_name, records in state.slots.items():
        active = []
        for r in records:
            if not r.is_active:
                continue
            active.append(
                {
                    "subject": r.subject,
                    "relation": r.relation,
                    "object": r.object,
                    "ttl": r.ttl,
                    "expires_at": _ttl_expires_at(r.ttl, r.created_at_datetime),
                }
            )
        if active:
            result[slot_name] = active

    return JSONResponse({"dialogue_id": dialogue_id, "slots": result})


@app.get("/dialogue/{dialogue_id}/graph/image")
def get_graph_image(dialogue_id: str) -> Response:
    """Return the memory graph as a PNG image (networkx + matplotlib)."""
    pipeline = _require_pipeline()
    state = pipeline.dst.get_state(dialogue_id)

    triplets: list[tuple[str, str, str, str]] = []
    for slot_name, records in state.slots.items():
        for r in records:
            if r.is_active:
                triplets.append((r.subject, r.relation, r.object, slot_name))

    buf = io.BytesIO()

    if not triplets:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5,
            0.5,
            f"No memory yet for '{dialogue_id}'",
            ha="center",
            va="center",
            fontsize=13,
            color="#888888",
        )
        ax.axis("off")
        fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")

    G = nx.DiGraph()
    edge_labels: dict[tuple, str] = {}
    slot_names = sorted({t[3] for t in triplets})
    colors = _slot_color_palette(slot_names)
    edge_colors: list[str] = []

    for subj, rel, obj, slot in triplets:
        G.add_node(subj)
        G.add_node(obj)
        key = (subj, obj)
        if key in edge_labels:
            edge_labels[key] += f"\n{rel}"
        else:
            G.add_edge(subj, obj)
            edge_labels[key] = rel
        edge_colors.append(colors.get(slot, "#888888"))

    n = len(G.nodes)
    fig_w = max(9, n * 1.4)
    fig_h = max(7, n * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    pos = nx.spring_layout(G, seed=42, k=2.5 / max(1, n**0.5))

    nx.draw_networkx_nodes(
        G, pos, node_color="#AED6F1", node_size=2000, ax=ax, alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=2.0,
        arrows=True,
        arrowsize=18,
        ax=ax,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=20,
        min_target_margin=20,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    legend_patches = [
        plt.matplotlib.patches.Patch(facecolor=colors[s], label=s) for s in slot_names
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_title(f"Memory: {dialogue_id}  ({len(triplets)} facts)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()

    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


@app.get("/dialogue/{dialogue_id}/graph/html")
def get_graph_html(dialogue_id: str) -> Response:
    """Return an interactive HTML visualization of the memory graph (pyvis).

    Open in a browser — nodes are draggable, edges are labelled with the relation,
    colour-coded by slot.  Empty memory returns a minimal HTML page.
    """
    from pyvis.network import Network

    pipeline = _require_pipeline()
    state = pipeline.dst.get_state(dialogue_id)

    triplets: list[tuple[str, str, str, str]] = []
    for slot_name, records in state.slots.items():
        for r in records:
            if r.is_active:
                triplets.append((r.subject, r.relation, r.object, slot_name))

    if not triplets:
        html = (
            "<!DOCTYPE html><html><body style='font-family:sans-serif;padding:40px'>"
            f"<h2>No memory yet for <code>{dialogue_id}</code></h2>"
            "</body></html>"
        )
        return Response(content=html, media_type="text/html; charset=utf-8")

    slot_names = sorted({t[3] for t in triplets})
    colors = _slot_color_palette(slot_names)

    net = Network(
        height="700px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
    )
    net.set_options(
        """{
        "physics": {"stabilization": {"iterations": 120}},
        "edges": {"smooth": {"type": "dynamic"}, "arrows": {"to": {"enabled": true}}},
        "nodes": {"font": {"size": 14}},
        "interaction": {"hover": true, "tooltipDelay": 100}
    }"""
    )

    seen_nodes: set[str] = set()
    for subj, rel, obj, slot in triplets:
        color = colors.get(slot, "#888888")
        if subj not in seen_nodes:
            net.add_node(
                subj,
                label=subj,
                title=f"slot: {slot}",
                color=color,
                size=20,
            )
            seen_nodes.add(subj)
        if obj not in seen_nodes:
            net.add_node(
                obj,
                label=obj,
                title=f"slot: {slot}",
                color=color,
                size=16,
            )
            seen_nodes.add(obj)
        net.add_edge(subj, obj, label=rel, title=rel, color=color)

    # Legend as an HTML overlay (pyvis doesn't have native legend support)
    legend_items = "".join(
        f'<span style="margin-right:12px">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{colors[s]};border-radius:50%;margin-right:4px"></span>'
        f"{s}</span>"
        for s in slot_names
    )
    title_bar = (
        f'<div style="position:fixed;top:0;left:0;right:0;z-index:999;'
        f"background:rgba(26,26,46,.9);color:#e0e0e0;padding:8px 16px;"
        f'font-family:sans-serif;font-size:13px">'
        f"<b>Memory graph:</b> {dialogue_id} &nbsp;|&nbsp; {len(triplets)} facts"
        f"&nbsp;&nbsp;{legend_items}"
        f"</div>"
    )

    raw_html = net.generate_html()
    # Inject title bar right after <body>
    html_out = raw_html.replace("<body>", f"<body>\n{title_bar}", 1)
    return Response(content=html_out, media_type="text/html; charset=utf-8")


@app.delete("/dialogue/{dialogue_id}")
def delete_dialogue(dialogue_id: str) -> JSONResponse:
    """Clear all memory (DST state + RAGU graph) for the given dialogue."""
    pipeline = _require_pipeline()
    lock = _state.lock_for(dialogue_id)
    with lock:
        pipeline.clear_memory(dialogue_id)
    logger.info("Dialogue cleared dialogue_id=%s", dialogue_id)
    return JSONResponse({"dialogue_id": dialogue_id, "status": "cleared"})


# ---------------------------------------------------------------------------
# Entry point (python api.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="GigaMemory REST API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="", help="Path to run_config_api.json")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.config:
        os.environ["GIGAMEMORY_CONFIG"] = args.config

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(Path(__file__).parent),
    )
