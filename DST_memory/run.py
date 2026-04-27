import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dst_memory.dotenv_loader import load_dst_memory_dotenv
from dst_memory.io_utils import iter_dialogue_messages, iter_user_messages, read_jsonl
from dst_memory.run_config_loader import (
    default_config_path,
    load_run_config,
    shared_section,
    subsection,
)


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


logger = logging.getLogger(__name__)


def _pre_parse_config_path(argv: list[str]) -> str | None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str, default=None)
    ns, _ = p.parse_known_args(argv)
    return ns.config


def build_pipeline(args: argparse.Namespace):
    from dst_memory import PipelineConfig
    from dst_memory.pipeline import DSTMemoryPipeline
    _ensure_local_ragu_import()
    from dst_memory.ragu_graph_processor import build_ragu_processor

    cfg = PipelineConfig(
        importance_model_path=args.importance_model_path,
        importance_threshold=args.importance_threshold,
        retrieval_top_k=args.retrieval_top_k,
        graph_top_k_records=args.graph_top_k_records,
        recent_history_pairs=args.recent_history_pairs,
        use_memory_gate=not args.disable_memory_gate,
        memory_gate_use_stub=args.memory_gate_use_stub,
        memory_strategy=args.memory_strategy,
        llm_mode=args.llm_mode,
        llm_api_url=args.llm_api_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_max_tokens=args.llm_max_tokens,
        openrouter_http_referer=args.openrouter_http_referer,
        openrouter_x_title=args.openrouter_x_title,
        llm_temperature=args.llm_temperature,
        slot_use_stub=args.slot_use_stub,
        slot_model_path=args.slot_model_path,
        slot_max_slots_per_message=args.slot_max_slots_per_message,
        use_ragu=True,
        ragu_embedder_model=getattr(args, "ragu_embedder_model", "deepvk/USER-bge-m3"),
        ragu_storage_path=getattr(args, "ragu_storage_path", ""),
    )

    logger.info(
        "Initializing RAGU backend embedder=%s storage=%s",
        cfg.ragu_embedder_model,
        cfg.ragu_storage_path or "<default>",
    )
    _kg, ragu_processor = build_ragu_processor(
        embedder_model=cfg.ragu_embedder_model,
        storage_path=cfg.ragu_storage_path or None,
    )
    return DSTMemoryPipeline(cfg, ragu_processor=ragu_processor)


def _ensure_local_ragu_import() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    local_ragu_root = repo_root / "RAGU"
    if not local_ragu_root.is_dir():
        return
    p = str(local_ragu_root.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def create_parser(
    *,
    resolved_config_path: str,
    shared: dict,
    pipeline_jsonl: dict,
    pipeline_interactive: dict,
) -> argparse.ArgumentParser:
    s = shared
    pj_cfg = pipeline_jsonl
    pi_cfg = pipeline_interactive

    parser = argparse.ArgumentParser(description="DST_memory runner")
    parser.add_argument("--config", type=str, default=resolved_config_path)
    parser.add_argument("--importance-model-path", type=str, default=s.get("importance_model_path", ""))
    parser.add_argument("--importance-threshold", type=float, default=float(s.get("importance_threshold", 0.5)))
    parser.add_argument("--retrieval-top-k", type=int, default=int(s.get("retrieval_top_k", 5)))
    parser.add_argument("--graph-top-k-records", type=int, default=int(s.get("graph_top_k_records", 20)))
    parser.add_argument("--recent-history-pairs", type=int, default=int(s.get("recent_history_pairs", 5)))
    parser.add_argument(
        "--disable-memory-gate",
        action="store_const",
        const=True,
        default=bool(s.get("disable_memory_gate", False)),
        help="Disable slot relevance gate for relevant_slots_full strategy.",
    )
    parser.add_argument(
        "--memory-gate-use-stub",
        action="store_const",
        const=True,
        default=bool(s.get("memory_gate_use_stub", False)),
    )
    parser.add_argument(
        "--memory-strategy",
        type=str,
        default=str(s.get("memory_strategy", "relevant_slots_full")),
        choices=["full_graph_json", "relevant_slots_full", "topk_graph_records"],
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        default=str(s.get("llm_mode", "openrouter")),
        choices=["stub", "local", "api", "openrouter"],
    )
    parser.add_argument("--llm-api-url", type=str, default=str(s.get("llm_api_url", "https://openrouter.ai/api/v1")))
    parser.add_argument("--llm-api-key", type=str, default=str(s.get("llm_api_key", "")))
    parser.add_argument("--llm-model", type=str, default=str(s.get("llm_model", "openai/gpt-oss-120b:free")))
    parser.add_argument("--llm-temperature", type=float, default=float(s.get("llm_temperature", 0.0)))
    parser.add_argument("--llm-max-tokens", type=int, default=int(s.get("llm_max_tokens", 1024)))
    parser.add_argument("--openrouter-http-referer", dest="openrouter_http_referer", type=str, default=str(s.get("openrouter_http_referer", "")))
    parser.add_argument("--openrouter-x-title", dest="openrouter_x_title", type=str, default=str(s.get("openrouter_x_title", "")))
    parser.add_argument("--no-final-llm", action="store_const", const=True, default=bool(s.get("no_final_llm", False)))
    parser.add_argument("--log-level", type=str, default=str(s.get("log_level", "INFO")))
    parser.add_argument("--slot-use-stub", action="store_const", const=True, default=bool(s.get("slot_use_stub", False)))
    parser.add_argument("--slot-model-path", type=str, default=str(s.get("slot_model_path", "Qwen/Qwen3-0.6B")))
    parser.add_argument("--slot-max-slots-per-message", type=int, default=int(s.get("slot_max_slots_per_message", 5)))
    parser.add_argument("--ragu-embedder-model", type=str, default=str(s.get("ragu_embedder_model", "deepvk/USER-bge-m3")))
    parser.add_argument("--ragu-storage-path", type=str, default=str(s.get("ragu_storage_path", "")))

    sub = parser.add_subparsers(required=True)
    m = sub.add_parser("module", help="Run single module")
    m_sub = m.add_subparsers(required=True)

    c = m_sub.add_parser("classifier", help="Run importance classifier")
    c.add_argument("--text", required=True, type=str)
    c.set_defaults(func=cmd_module_classifier)

    d = m_sub.add_parser("dst", help="Run DST upsert")
    d.add_argument("--dialogue-id", required=True, type=str)
    d.add_argument("--text", required=True, type=str)
    d.set_defaults(func=cmd_module_dst)

    op = m_sub.add_parser("openrouter-ping", help="Single ping to OpenRouter")
    op.add_argument("--prompt", type=str, default="Ответь одним словом: работает.")
    op.set_defaults(func=cmd_module_openrouter_ping)

    tj = m_sub.add_parser("triplet-json-test", help="Run per-slot triplet extraction from JSON payload")
    tj.add_argument("--json-path", type=str, default="DST_memory/triplet_test_payload.json")
    tj.set_defaults(func=cmd_module_triplet_json_test)

    tjb = m_sub.add_parser("triplet-json-batch-test", help="Run triplet extraction for batch JSON payload")
    tjb.add_argument("--json-path", type=str, default="DST_memory/triplet_test_payloads.json")
    tjb.add_argument("--output-path", type=str, default="DST_memory/triplet_test_results.json")
    tjb.set_defaults(func=cmd_module_triplet_json_batch_test)

    p = sub.add_parser("pipeline", help="Run pipeline")
    p_sub = p.add_subparsers(required=True)

    test_parser = p_sub.add_parser("test", help="Test mode over jsonl dataset")
    test_parser.add_argument("--dataset-path", type=str, default=pj_cfg.get("dataset_path"))
    test_parser.add_argument("--output-path", type=str, default=pj_cfg.get("output_path"))
    test_parser.set_defaults(func=cmd_pipeline_test_jsonl)

    inf = p_sub.add_parser("inference", help="Inference mode")
    inf_sub = inf.add_subparsers(required=True)

    pi_parser = inf_sub.add_parser("interactive", help="Interactive inference")
    pi_parser.add_argument("--dialogue-id", dest="dialogue_id", type=str, default=pi_cfg.get("dialogue_id", "interactive"))
    pi_parser.set_defaults(func=cmd_pipeline_inference_interactive)

    ps_parser = inf_sub.add_parser("single-turn", help="Single turn inference")
    ps_parser.add_argument("--dialogue-id", type=str, default="single_turn")
    ps_parser.add_argument("--message", required=True, type=str)
    ps_parser.set_defaults(func=cmd_pipeline_inference_single_turn)
    return parser


def cmd_module_classifier(args: argparse.Namespace) -> None:
    from dst_memory.classifier import ImportanceClassifier

    model = ImportanceClassifier(model_path=args.importance_model_path, threshold=args.importance_threshold)
    print(json.dumps(model.predict(args.text), ensure_ascii=False, indent=2))


def cmd_module_dst(args: argparse.Namespace) -> None:
    from dst_memory.dst_manager import DSTManager
    from dst_memory.serving import LocalHFServing
    from dst_memory.slot_select_client import SlotSelectClient
    from dst_memory.triplet_client import TripletExtractionClient

    slot_serving = None
    if not args.slot_use_stub:
        slot_serving = LocalHFServing(args.slot_model_path)
    triplet_extractor = TripletExtractionClient(
        use_stub=args.slot_use_stub,
        serving=slot_serving,
        max_triplets=max(6, int(args.slot_max_slots_per_message) * 3),
        max_retries=1,
    )
    slot_selector = SlotSelectClient(
        use_stub=args.slot_use_stub,
        serving=slot_serving,
        max_slots=int(args.slot_max_slots_per_message),
        max_retries=1,
    )
    dst = DSTManager(triplet_extractor=triplet_extractor, slot_selector=slot_selector, single_pass_fallback=True)
    print(json.dumps([asdict(x) for x in dst.upsert_from_message(args.dialogue_id, args.text)], ensure_ascii=False, indent=2))


def cmd_module_openrouter_ping(args: argparse.Namespace) -> None:
    from dst_memory.llm_client import FinalLLMClient

    client = FinalLLMClient(
        mode="openrouter",
        api_url=args.llm_api_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        temperature=float(args.llm_temperature),
        max_tokens=min(int(args.llm_max_tokens), 256),
        http_referer=args.openrouter_http_referer,
        x_title=args.openrouter_x_title,
    )
    print(client.generate(question=args.prompt, memory_context={}, recent_pairs=[]))


def cmd_module_triplet_json_test(args: argparse.Namespace) -> None:
    from dst_memory.ontology import DEFAULT_USER_SLOTS
    from dst_memory.serving import LocalHFServing
    from dst_memory.triplet_client import TripletExtractionClient

    with open(args.json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    msg = str(payload.get("message", "")).strip()
    slot = str(payload.get("slot", "")).strip().upper()
    if not msg:
        raise SystemExit("JSON payload must include non-empty 'message'")
    if slot not in set(DEFAULT_USER_SLOTS.slot_names):
        raise SystemExit(f"Invalid slot '{slot}'. Allowed: {', '.join(DEFAULT_USER_SLOTS.slot_names)}")
    serving = LocalHFServing(args.slot_model_path)
    client = TripletExtractionClient(
        use_stub=False,
        serving=serving,
        max_triplets=max(6, int(args.slot_max_slots_per_message) * 3),
        max_retries=1,
    )
    triplets = client.extract_for_slot(msg, slot)
    out = {"slot": slot, "message": msg, "triplets": [{"subject": t.subject, "relation": t.relation, "object": t.object} for t in triplets]}
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_module_triplet_json_batch_test(args: argparse.Namespace) -> None:
    from dst_memory.ontology import DEFAULT_USER_SLOTS
    from dst_memory.serving import LocalHFServing
    from dst_memory.triplet_client import TripletExtractionClient

    with open(args.json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Batch JSON must contain non-empty 'cases' list")
    allowed = set(DEFAULT_USER_SLOTS.slot_names)
    serving = LocalHFServing(args.slot_model_path)
    client = TripletExtractionClient(
        use_stub=False,
        serving=serving,
        max_triplets=max(6, int(args.slot_max_slots_per_message) * 3),
        max_retries=1,
    )
    results = []
    for idx, case in enumerate(cases, start=1):
        cid = str(case.get("id", f"case_{idx}"))
        msg = str(case.get("message", "")).strip()
        slot = str(case.get("slot", "")).strip().upper()
        if not msg or slot not in allowed:
            results.append({"id": cid, "slot": slot, "message": msg, "error": "invalid_case", "triplets": []})
            continue
        triplets = client.extract_for_slot(msg, slot)
        results.append({"id": cid, "slot": slot, "message": msg, "triplets": [{"subject": t.subject, "relation": t.relation, "object": t.object} for t in triplets]})
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output_path}")


def cmd_pipeline_test_jsonl(args: argparse.Namespace) -> None:
    if not args.dataset_path or not args.output_path:
        raise SystemExit("pipeline test requires --dataset-path and --output-path")
    pipeline = build_pipeline(args)
    rows = read_jsonl(args.dataset_path)
    results_logs = []
    results_compact = []
    for row in rows:
        dialogue_id = str(row.get("id"))
        question = row.get("question", "")
        write_logs = []
        for msg in iter_user_messages(row):
            write_logs.append(pipeline.write_to_memory(dialogue_id=dialogue_id, message=msg))

        pending_user = None
        for msg in iter_dialogue_messages(row):
            if msg.role == "user":
                pending_user = msg.content
            elif msg.role == "assistant" and pending_user:
                pipeline.add_recent_pair(dialogue_id, pending_user, msg.content)
                pending_user = None

        answer = (
            pipeline.answer_without_final_llm(dialogue_id=dialogue_id, question=question)
            if args.no_final_llm
            else pipeline.answer(dialogue_id=dialogue_id, question=question)
        )
        results_logs.append({"dialogue_id": dialogue_id, "question": question, "write_logs": write_logs, "answer": answer})
        compact_answer = answer if isinstance(answer, dict) else pipeline.answer_without_final_llm(dialogue_id=dialogue_id, question=question)
        results_compact.append(
            {
                "dialogue_id": dialogue_id,
                "question": question,
                "use_memory": compact_answer.get("use_memory"),
                "retrieved": compact_answer.get("retrieved", []),
                "memory_slots": compact_answer.get("memory_slots", []),
                "memory_context": compact_answer.get("memory_context_for_final_llm"),
                "recent_pairs": compact_answer.get("recent_pairs", []),
            }
        )
        pipeline.clear_memory(dialogue_id)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results_compact, f, ensure_ascii=False, indent=2)
    logs_output_path = args.output_path.removesuffix(".json") + "_logs.json"
    with open(logs_output_path, "w", encoding="utf-8") as f:
        json.dump(results_logs, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output_path}")
    print(f"Saved logs: {logs_output_path}")


def cmd_pipeline_inference_interactive(args: argparse.Namespace) -> None:
    from dst_memory.models import Message

    pipeline = build_pipeline(args)
    did = args.dialogue_id
    print("Inference mode. Commands: /clear, /exit")
    while True:
        raw = input("user> ").strip()
        if not raw:
            continue
        if raw == "/exit":
            break
        if raw == "/clear":
            pipeline.clear_memory(did)
            print("memory cleared")
            continue
        log = pipeline.write_to_memory(did, Message(role="user", content=raw))
        if args.no_final_llm:
            out = pipeline.answer_without_final_llm(did, raw)
            print(json.dumps({"write_log": log, "answer": out}, ensure_ascii=False, indent=2))
            continue
        answer = pipeline.answer(did, raw)
        pipeline.add_recent_pair(did, raw, answer)
        print(answer)


def cmd_pipeline_inference_single_turn(args: argparse.Namespace) -> None:
    from dst_memory.models import Message

    msg = args.message.strip()
    if not msg:
        raise SystemExit("single-turn requires non-empty --message")
    pipeline = build_pipeline(args)
    log = pipeline.write_to_memory(args.dialogue_id, Message(role="user", content=msg))
    if args.no_final_llm:
        out = pipeline.answer_without_final_llm(args.dialogue_id, msg)
        print(json.dumps({"write_log": log, "answer": out}, ensure_ascii=False, indent=2))
        return
    answer = pipeline.answer(args.dialogue_id, msg)
    pipeline.add_recent_pair(args.dialogue_id, msg, answer)
    print(json.dumps({"write_log": log, "answer": answer}, ensure_ascii=False, indent=2))


def main() -> None:
    load_dst_memory_dotenv()
    argv = sys.argv[1:]
    config_path_opt = _pre_parse_config_path(argv)
    env_path = os.environ.get("DST_MEMORY_CONFIG")
    load_path: str | Path | None = config_path_opt or env_path
    try:
        file_cfg = load_run_config(load_path)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nCreate DST_memory/run_config.json or pass --config <path>.") from e
    resolved = Path(load_path) if load_path else default_config_path()
    resolved = resolved.resolve()
    parser = create_parser(
        resolved_config_path=str(resolved),
        shared=shared_section(file_cfg),
        pipeline_jsonl=subsection(file_cfg, "pipeline_jsonl"),
        pipeline_interactive=subsection(file_cfg, "pipeline_interactive"),
    )
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    logger.info("DST_memory run started config=%s", resolved)
    args.func(args)


if __name__ == "__main__":
    main()
