"""GigaMemory FastAPI REST server.

Endpoints:
  POST   /dialogue/{dialogue_id}/message        — send message, get LLM answer
  GET    /dialogue/{dialogue_id}/graph          — full memory graph (JSON)
  GET    /dialogue/{dialogue_id}/graph_short    — compact: active triplets + TTL deadline only
  GET    /dialogue/{dialogue_id}/graph/image    — memory graph as PNG
  GET    /dialogue/{dialogue_id}/graph/html     — interactive memory graph (pyvis HTML)
  DELETE /dialogue/{dialogue_id}               — reset dialogue memory
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from dst_memory.utils.dotenv_loader import load_dst_memory_dotenv

load_dst_memory_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared app state
# ---------------------------------------------------------------------------


class _State:
    pipeline: Any = None
    session_dir: str = ""
    ragu_tmpdir: str = ""  # ephemeral RAGU storage, removed on shutdown
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

    # RAGU in the long-lived API server is embedder-only: the final-LLM read path
    # builds memory context from the per-dialogue DST state (full_graph_json),
    # never from RAGU's graph/vector search. So we deliberately do NOT persist the
    # RAGU index into api_sessions/ (which is reserved for per-dialogue state.json)
    # — a single shared store there would mix every dialogue's triplets, grow
    # unbounded, and survive /forget. Instead we hand RAGU an ephemeral temp dir
    # that is removed on shutdown. (RAGU's KV/vector backends are file-based, so a
    # folder must exist; it just lives outside the project and is throwaway.)
    ragu_storage = tempfile.mkdtemp(prefix="gigamemory_ragu_")
    _state.ragu_tmpdir = ragu_storage

    cfg = PipelineConfig(
        importance_model_path=str(s.get("importance_model_path", "")),
        importance_threshold=float(s.get("importance_threshold", 0.25)),
        retrieval_top_k=int(s.get("retrieval_top_k", 5)),
        graph_top_k_records=int(s.get("graph_top_k_records", 20)),
        recent_history_pairs=int(s.get("recent_history_pairs", 5)),
        use_memory_gate=not bool(s.get("disable_memory_gate", False)),
        memory_gate_use_stub=bool(s.get("memory_gate_use_stub", False)),
        memory_strategy=str(s.get("memory_strategy", "relevant_slots_full")),
        relevant_slots_always_include_identity=bool(
            s.get("relevant_slots_always_include_identity", False)
        ),
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
        slot_llm_mode=str(s.get("slot_llm_mode", "local")),
        slot_llm_api_url=str(
            s.get("slot_llm_api_url", "") or "http://localhost:8001/v1"
        ),
        slot_llm_api_key=str(s.get("slot_llm_api_key", "EMPTY")),
        slot_select_max_tokens=int(s.get("slot_select_max_tokens", 220)),
        triplet_extract_max_tokens=int(s.get("triplet_extract_max_tokens", 512)),
        conflict_max_tokens=int(s.get("conflict_max_tokens", 256)),
        deletion_max_tokens=int(s.get("deletion_max_tokens", 256)),
        memory_gate_max_tokens=int(s.get("memory_gate_max_tokens", 200)),
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
    if _state.ragu_tmpdir:
        shutil.rmtree(_state.ragu_tmpdir, ignore_errors=True)
        logger.info("Removed ephemeral RAGU storage %s", _state.ragu_tmpdir)


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
    # Optional per-request language ("ru" | "en"). When omitted the server's
    # configured prompt_language is used. Applies to the whole request: memory
    # extraction (write), the read-path gate, and the final-LLM answer. A
    # dialogue's graph must stay single-language — the caller (bot) enforces
    # this by clearing memory when the user switches language.
    prompt_language: str | None = None


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


# ---------------------------------------------------------------------------
# Graph visualization — replicates RAGU's knowledge-graph rendering logic
# (scripts/visualize_knowledge_graph.py) but written standalone to avoid any
# coupling with the RAGU package: nodes coloured by slot, sized by degree;
# edges coloured by TTL; per-slot + TTL legends.
# ---------------------------------------------------------------------------

# Entities are scoped per slot (see _build_graph_model): every node belongs to
# exactly one slot and is coloured by it. Fallback if a colour is ever missing.
_DEFAULT_NODE_COLOR = "#BDC3C7"

# Separator for slot-scoped node IDs.
_NODE_ID_SEP = "\x00"

# TTL → edge colour (green = long-lived, amber = medium, red = short-lived).
_TTL_EDGE_COLORS: dict[str, str] = {
    "inf": "#27AE60",
    "1y": "#2ECC71",
    "6m": "#A9DFBF",
    "3m": "#F39C12",
    "1m": "#E67E22",
    "3w": "#E67E22",
    "2w": "#E67E22",
    "10d": "#E74C3C",
    "3d": "#C0392B",
    "1d": "#922B21",
}
_TTL_DEFAULT_EDGE_COLOR = "#95A5A6"

_TTL_DISPLAY: dict[str, str] = {
    "inf": "бессрочно",
    "1y": "1 год",
    "6m": "6 месяцев",
    "3m": "3 месяца",
    "1m": "1 месяц",
    "3w": "3 недели",
    "2w": "2 недели",
    "10d": "10 дней",
    "3d": "3 дня",
    "1d": "1 день",
}


def _ttl_edge_color(ttl: str | None) -> str:
    if not ttl:
        return _TTL_DEFAULT_EDGE_COLOR
    return _TTL_EDGE_COLORS.get(ttl, _TTL_DEFAULT_EDGE_COLOR)


def _ttl_display(ttl: str | None) -> str:
    if not ttl:
        return "не задан"
    return _TTL_DISPLAY.get(ttl, ttl)


def _generate_distinct_colors(n: int) -> list[str]:
    """Golden-ratio HSL palette — same approach RAGU uses for entity types."""
    import colorsys

    colors: list[str] = []
    golden_ratio = 0.618033988749895
    for i in range(n):
        hue = (i * golden_ratio) % 1.0
        saturation = 0.65 + (i % 3) * 0.1
        lightness = 0.5 + (i % 2) * 0.1
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(
            f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
        )
    return colors


def _build_slot_color_map(slot_names: set[str]) -> dict[str, str]:
    """Map each slot to a distinct colour (golden-ratio palette, like RAGU)."""
    ordered = sorted(slot_names)
    palette = _generate_distinct_colors(len(ordered))
    return {slot: palette[i] for i, slot in enumerate(ordered)}


def _collect_active_facts(state: Any) -> list[tuple[str, str, str, str, str, str]]:
    """Flatten active records → (subject, relation, object, slot, ttl, created)."""
    facts: list[tuple[str, str, str, str, str, str]] = []
    for slot_name, records in state.slots.items():
        for r in records:
            if r.is_active:
                facts.append(
                    (
                        r.subject,
                        r.relation,
                        r.object,
                        slot_name,
                        getattr(r, "ttl", "inf"),
                        getattr(r, "created_at_datetime", ""),
                    )
                )
    return facts


def _node_id(slot: str, entity: str) -> str:
    return f"{slot}{_NODE_ID_SEP}{entity}"


def _layout_disjoint(
    graph: nx.DiGraph,
) -> tuple[dict[str, tuple[float, float]], int, int]:
    """Lay out each connected component (= one slot) in its own grid cell.

    A plain spring_layout piles disconnected components on top of each other;
    placing each component in a separate cell keeps the slots visually apart.
    Returns (positions, cols, rows).
    """
    import math

    components = sorted(
        nx.connected_components(graph.to_undirected()), key=len, reverse=True
    )
    ncomp = max(1, len(components))
    cols = max(1, math.ceil(math.sqrt(ncomp)))
    rows = max(1, math.ceil(ncomp / cols))
    # Spread nodes well within each cell (scale) and keep cells far enough apart
    # (spacing) that neighbouring slot clusters never touch. spacing must stay
    # comfortably above 2*scale so the cells cannot overlap.
    scale = 1.6
    spacing = 6.0

    pos: dict[str, tuple[float, float]] = {}
    for idx, comp in enumerate(components):
        sub = graph.subgraph(comp)
        sub_pos = nx.spring_layout(sub, seed=42, k=2.5, scale=scale)
        cx = (idx % cols) * spacing
        cy = -(idx // cols) * spacing
        for node, (x, y) in sub_pos.items():
            pos[node] = (cx + x, cy + y)
    return pos, cols, rows


def _build_graph_model(
    facts: list[tuple[str, str, str, str, str, str]],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, int],
    list[tuple[str, str, str, str, str]],
]:
    """Entity-relation model à la RAGU, with per-slot node scoping.

    Crucially, entities are NOT shared across slots: each slot gets its OWN copy
    of every entity (including "пользователь"), keyed by an id of ``slot\\0entity``.
    This makes the slots completely disjoint subgraphs that never intersect — the
    same way RAGU renders a separate "user" hub per slot/cluster.

    Returns (node_label, node_slot, degree, edges):
      - node_label: node_id → display label (the bare entity name).
      - node_slot:  node_id → owning slot (drives node colour / group).
      - degree:     node_id → incident-edge count (drives node size).
      - edges:      (subject_id, object_id, relation, slot, ttl).
    """
    node_label: dict[str, str] = {}
    node_slot: dict[str, str] = {}
    degree: dict[str, int] = {}
    edges: list[tuple[str, str, str, str, str]] = []
    for subj, rel, obj, slot, ttl, _created in facts:
        s_id = _node_id(slot, subj)
        o_id = _node_id(slot, obj)
        node_label[s_id] = subj
        node_label[o_id] = obj
        node_slot[s_id] = slot
        node_slot[o_id] = slot
        degree[s_id] = degree.get(s_id, 0) + 1
        degree[o_id] = degree.get(o_id, 0) + 1
        edges.append((s_id, o_id, rel, slot, ttl))
    return node_label, node_slot, degree, edges


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
                pipeline.write_to_memory(
                    dialogue_id, msg, prompt_language=req.prompt_language
                )
                _save_session(dialogue_id)
            except Exception as e:
                logger.error(
                    "Background write failed dialogue_id=%s: %s", dialogue_id, e
                )

        t = threading.Thread(target=_bg_write, daemon=True)
        t.start()
        answer = pipeline.answer(
            dialogue_id, req.content, prompt_language=req.prompt_language
        )
        pipeline.add_recent_pair(dialogue_id, req.content, answer)
    else:
        lock = _state.lock_for(dialogue_id)
        with lock:
            pipeline.write_to_memory(
                dialogue_id, msg, prompt_language=req.prompt_language
            )
            answer = pipeline.answer(
                dialogue_id, req.content, prompt_language=req.prompt_language
            )
            pipeline.add_recent_pair(dialogue_id, req.content, answer)
            _save_session(dialogue_id)

    return MessageResponse(dialogue_id=dialogue_id, answer=answer)


@app.post("/dialogue/{dialogue_id}/answer", response_model=MessageResponse)
def post_answer(dialogue_id: str, req: MessageRequest) -> MessageResponse:
    """Generate the assistant answer only — no memory write.

    Lets a client drive the answer and the memory write as two independent
    phases (e.g. to show separate progress to the user). Mirrors the
    parallel_write path, where the answer does not wait for the current message
    to be written to memory.
    """
    pipeline = _require_pipeline()
    answer = pipeline.answer(
        dialogue_id, req.content, prompt_language=req.prompt_language
    )
    pipeline.add_recent_pair(dialogue_id, req.content, answer)
    return MessageResponse(dialogue_id=dialogue_id, answer=answer)


@app.post("/dialogue/{dialogue_id}/remember")
def post_remember(dialogue_id: str, req: MessageRequest) -> dict:
    """Write the user message to memory only — no answer generated."""
    from dst_memory.core.models import Message

    pipeline = _require_pipeline()
    msg = Message(role="user", content=req.content)
    result = pipeline.write_to_memory(
        dialogue_id, msg, prompt_language=req.prompt_language
    )
    _save_session(dialogue_id)
    return {"dialogue_id": dialogue_id, "saved": bool(result.get("saved", False))}


@app.get("/dialogue/{dialogue_id}/graph")
def get_graph(dialogue_id: str) -> JSONResponse:
    """Return the full memory graph as JSON (active facts with all metadata)."""
    pipeline = _require_pipeline()
    slots = pipeline.dst.slots_with_messages(dialogue_id)
    return JSONResponse({"dialogue_id": dialogue_id, "slots": slots})


@app.get("/dialogue/{dialogue_id}/context")
def get_context(dialogue_id: str) -> JSONResponse:
    """Return the recent conversation turns passed verbatim to the final LLM.

    These are what the model "remembers" directly from the dialogue window, as
    opposed to facts retrieved from the memory graph (see /graph_short). ``limit``
    is the configured number of pairs kept in that window.
    """
    pipeline = _require_pipeline()
    pairs = pipeline.recent_pairs(dialogue_id)
    limit = int(getattr(pipeline.config, "recent_history_pairs", 0))
    return JSONResponse(
        {"dialogue_id": dialogue_id, "recent_pairs": pairs, "limit": limit}
    )


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
    """Return the memory graph as a PNG image.

    Same logical model as RAGU's interactive view: one node per entity, coloured
    by its slot and sized by degree; edges coloured by TTL and labelled with the
    relation. Rendered statically with networkx + matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    pipeline = _require_pipeline()
    state = pipeline.dst.get_state(dialogue_id)
    facts = _collect_active_facts(state)

    buf = io.BytesIO()

    if not facts:
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

    node_label, node_slot, degree, edges = _build_graph_model(facts)
    slot_color = _build_slot_color_map(set(node_slot.values()))

    G = nx.DiGraph()
    for node in node_slot:
        G.add_node(node)

    # Collapse parallel relations between the same pair into one labelled edge;
    # edge colour comes from the TTL of that relation. Identical relations are
    # de-duplicated so a repeated fact does not print "имя\nимя".
    edge_label: dict[tuple[str, str], str] = {}
    drawn_edges: list[tuple[str, str]] = []
    edge_colors: list[str] = []
    for s_id, o_id, rel, _slot, ttl in edges:
        key = (s_id, o_id)
        if key in edge_label:
            if rel not in edge_label[key].split("\n"):
                edge_label[key] += f"\n{rel}"
        else:
            edge_label[key] = rel
            drawn_edges.append(key)
            G.add_edge(s_id, o_id)
            edge_colors.append(_ttl_edge_color(ttl))

    pos, cols, rows = _layout_disjoint(G)
    fig_w = max(9, cols * 5.5)
    fig_h = max(7, rows * 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    max_deg = max(degree.values()) if degree else 1
    node_colors = [
        slot_color.get(node_slot[node], _DEFAULT_NODE_COLOR) for node in G.nodes
    ]
    node_sizes = [800 + 2200 * (degree.get(node, 1) / max_deg) for node in G.nodes]
    labels = {node: node_label[node] for node in G.nodes}

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
        alpha=0.95,
        edgecolors="#333333",
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=drawn_edges,
        edge_color=edge_colors,
        width=2.0,
        arrows=True,
        arrowsize=18,
        ax=ax,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=20,
        min_target_margin=20,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label, font_size=8, ax=ax)

    # Two legends: slot → node colour, TTL → edge colour.
    slot_patches = [
        mpatches.Patch(facecolor=slot_color[s], label=s) for s in sorted(slot_color)
    ]
    ttls_present = sorted({ttl for *_rest, ttl in edges})
    ttl_patches = [
        mpatches.Patch(
            facecolor=_ttl_edge_color(t), label=f"TTL {t} ({_ttl_display(t)})"
        )
        for t in ttls_present
    ]
    leg1 = ax.legend(
        handles=slot_patches,
        loc="upper left",
        fontsize=8,
        framealpha=0.85,
        title="Слоты (узлы)",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=ttl_patches,
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
        title="TTL (рёбра)",
    )

    ax.set_title(f"Memory: {dialogue_id}  ({len(facts)} facts)", fontsize=11)
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
    facts = _collect_active_facts(state)

    if not facts:
        html = (
            "<!DOCTYPE html><html><body style='font-family:sans-serif;padding:40px'>"
            f"<h2>No memory yet for <code>{dialogue_id}</code></h2>"
            "</body></html>"
        )
        return Response(content=html, media_type="text/html; charset=utf-8")

    node_label, node_slot, degree, edges = _build_graph_model(facts)
    slot_color = _build_slot_color_map(set(node_slot.values()))

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#222222",
        font_color="white",
    )
    net.set_options(
        """
    {
        "nodes": {"font": {"size": 14, "face": "arial"}, "borderWidth": 2, "borderWidthSelected": 4},
        "edges": {"color": {"inherit": false}, "smooth": {"type": "continuous", "forceDirection": "none"}, "font": {"size": 10, "align": "middle"}},
        "physics": {"enabled": true, "solver": "forceAtlas2Based", "forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 150, "springConstant": 0.08, "damping": 0.4}, "stabilization": {"enabled": true, "iterations": 200, "updateInterval": 25}},
        "interaction": {"hover": true, "tooltipDelay": 100, "hideEdgesOnDrag": true}
    }
    """
    )

    # Nodes: one per (slot, entity), coloured by slot, sized by degree, grouped
    # by slot. Because ids are slot-scoped, every slot keeps its own "user" node
    # and the slots render as completely disjoint clusters.
    max_deg = max(degree.values()) if degree else 1
    min_size, max_size = 15.0, 50.0
    for node, slot in node_slot.items():
        deg = degree.get(node, 1)
        size = min_size + (max_size - min_size) * (deg / max_deg)
        color = slot_color.get(slot, _DEFAULT_NODE_COLOR)
        label = node_label[node]
        title = (
            f"<b>Сущность:</b> {label}<br>"
            f"<b>Слот:</b> {slot}<br>"
            f"<b>Связей:</b> {deg}"
        )
        net.add_node(
            node,
            label=label,
            title=title,
            size=size,
            color=color,
            shape="dot",
            group=slot,
        )

    # Edges: coloured by TTL, labelled with the relation. Skip exact duplicate
    # (subject, object, relation) edges so a repeated fact is not drawn twice.
    seen_edges: set[tuple[str, str, str]] = set()
    for s_id, o_id, rel, slot, ttl in edges:
        if (s_id, o_id, rel) in seen_edges:
            continue
        seen_edges.add((s_id, o_id, rel))
        title = (
            f"<b>Отношение:</b> {rel}<br>"
            f"<b>Слот:</b> {slot}<br>"
            f"<b>TTL:</b> {ttl} ({_ttl_display(ttl)})"
        )
        net.add_edge(s_id, o_id, label=rel, title=title, color=_ttl_edge_color(ttl))

    # Legend overlay: slot colours (nodes) + TTL colours (edges) + stats.
    slot_legend = "".join(
        f'<div style="display:flex;align-items:center;margin:3px 0;">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{slot_color[s]};border-radius:50%;margin-right:8px;"></span>'
        f"<span>{s}</span></div>"
        for s in sorted(slot_color)
    )
    ttls_present = sorted({ttl for *_rest, ttl in edges})
    ttl_legend = "".join(
        f'<div style="margin:2px 0;">'
        f'<span style="color:{_ttl_edge_color(t)}">&#9632;</span> '
        f"{t} — {_ttl_display(t)}</div>"
        for t in ttls_present
    )
    legend_html = (
        '<div id="legend" style="position:absolute;top:10px;right:10px;'
        "background:rgba(0,0,0,0.85);padding:15px;border-radius:8px;color:white;"
        "font-family:Arial,sans-serif;font-size:12px;max-height:600px;"
        'overflow-y:auto;z-index:1000;min-width:200px;">'
        f'<div style="font-weight:bold;margin-bottom:8px;font-size:14px;">'
        f"Memory: {dialogue_id}</div>"
        '<div style="font-weight:bold;margin-bottom:6px;">Слоты (цвет узлов)</div>'
        f"{slot_legend}"
        '<hr style="border-color:#444;margin:10px 0;">'
        '<div style="font-weight:bold;margin-bottom:6px;">TTL (цвет рёбер)</div>'
        f"{ttl_legend}"
        '<hr style="border-color:#444;margin:10px 0;">'
        f'<div style="font-size:11px;color:#aaa;">'
        f"<div>Узлов: {len(node_slot)}</div>"
        f"<div>Рёбер: {len(edges)}</div>"
        f"<div>Фактов: {len(facts)}</div>"
        '<div style="margin-top:5px;">Размер узла = число связей</div>'
        "<div>Цвет ребра = TTL</div></div>"
        "</div>"
    )

    raw_html = net.generate_html()
    html_out = raw_html.replace("</body>", legend_html + "</body>", 1)
    return Response(content=html_out, media_type="text/html; charset=utf-8")


@app.delete("/dialogue/{dialogue_id}")
def delete_dialogue(dialogue_id: str) -> JSONResponse:
    """Clear the per-dialogue DST memory state (the API's source of truth).

    RAGU here is ephemeral, embedder-only scratch storage that no read path
    queries, so there is nothing dialogue-scoped to clear in it.
    """
    pipeline = _require_pipeline()
    lock = _state.lock_for(dialogue_id)
    with lock:
        pipeline.clear_memory(dialogue_id)
    logger.info("Dialogue cleared dialogue_id=%s", dialogue_id)
    return JSONResponse({"dialogue_id": dialogue_id, "status": "cleared"})


@app.get("/health")
def health() -> JSONResponse:
    """Liveness probe used by Docker healthcheck and load balancers."""
    return JSONResponse(
        {"status": "ok", "pipeline_loaded": _state.pipeline is not None}
    )


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
