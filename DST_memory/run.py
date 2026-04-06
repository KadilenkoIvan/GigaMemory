import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dst_memory.dotenv_loader import load_dst_memory_dotenv
from dst_memory.io_utils import iter_user_messages, read_jsonl
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

    cfg = PipelineConfig(
        importance_model_path=args.importance_model_path,
        importance_threshold=args.importance_threshold,
        retrieval_top_k=args.retrieval_top_k,
        use_memory_gate=not args.disable_memory_gate,
        memory_gate_use_stub=args.memory_gate_use_stub,
        memory_context_source=args.memory_context_source,
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
    )
    return DSTMemoryPipeline(cfg)


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
    parser.add_argument(
        "--config",
        type=str,
        default=resolved_config_path,
        help="Path to run_config.json (values here are defaults; CLI overrides).",
    )
    parser.add_argument(
        "--importance-model-path",
        type=str,
        default=s.get("importance_model_path", ""),
    )
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=float(s.get("importance_threshold", 0.5)),
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=int(s.get("retrieval_top_k", 5)),
    )
    parser.add_argument(
        "--disable-memory-gate",
        action="store_const",
        const=True,
        default=bool(s.get("disable_memory_gate", False)),
        help="Отключить шлюз «нужна ли память»: slots — все записи всех слотов; vector — top-k из вектора без отбора LLM.",
    )
    parser.add_argument(
        "--memory-gate-use-stub",
        action="store_const",
        const=True,
        default=bool(s.get("memory_gate_use_stub", False)),
        help="Не вызывать локальную LLM для отбора слотов (эвристика по маркерам в тексте).",
    )
    parser.add_argument(
        "--memory-context-source",
        type=str,
        default=str(s.get("memory_context_source", "slots")),
        choices=["slots", "vector"],
        help='Откуда брать текст памяти для финальной LLM: "slots" — записи выбранных слотов; '
        '"vector" — top retrieval_top_k из векторного индекса.',
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        default=str(s.get("llm_mode", "stub")),
        choices=["stub", "local", "api", "openrouter"],
        help="Final answer: stub | local (TODO) | openrouter | api (OpenAI-compatible chat).",
    )
    parser.add_argument(
        "--llm-api-url",
        type=str,
        default=str(s.get("llm_api_url", "https://openrouter.ai/api/v1")),
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=str(s.get("llm_api_key", "")),
        help="API key (or set OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=str(s.get("llm_model", "openai/gpt-oss-120b:free")),
        help="Model id, e.g. openai/gpt-oss-120b:free, qwen/qwen3.6-plus:free, qwen/qwen3-coder:free.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=float(s.get("llm_temperature", 0.0)),
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=int(s.get("llm_max_tokens", 1024)),
    )
    parser.add_argument(
        "--openrouter-http-referer",
        dest="openrouter_http_referer",
        type=str,
        default=str(s.get("openrouter_http_referer", "")),
        help="Optional HTTP-Referer header for OpenRouter attribution.",
    )
    parser.add_argument(
        "--openrouter-x-title",
        dest="openrouter_x_title",
        type=str,
        default=str(s.get("openrouter_x_title", "")),
        help="Optional X-OpenRouter-Title header.",
    )
    parser.add_argument(
        "--no-final-llm",
        action="store_const",
        const=True,
        default=bool(s.get("no_final_llm", False)),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=str(s.get("log_level", "INFO")),
    )
    parser.add_argument(
        "--slot-use-stub",
        action="store_const",
        const=True,
        default=bool(s.get("slot_use_stub", False)),
    )
    parser.add_argument(
        "--slot-model-path",
        type=str,
        default=str(s.get("slot_model_path", "Qwen/Qwen3-0.6B")),
    )
    parser.add_argument(
        "--slot-max-slots-per-message",
        type=int,
        default=int(s.get("slot_max_slots_per_message", 5)),
    )

    sub = parser.add_subparsers(required=True)

    m = sub.add_parser("module", help="Run single module")
    m_sub = m.add_subparsers(required=True)

    c = m_sub.add_parser("classifier", help="Run importance classifier")
    c.add_argument("--text", required=True, type=str)
    c.set_defaults(func=cmd_module_classifier)

    d = m_sub.add_parser("dst", help="Run DST upsert stub")
    d.add_argument("--dialogue-id", required=True, type=str)
    d.add_argument("--text", required=True, type=str)
    d.set_defaults(func=cmd_module_dst)

    v = m_sub.add_parser("vector", help="Run vector store module")
    v.add_argument("--dialogue-id", required=True, type=str)
    v.add_argument("--query", required=True, type=str)
    v.add_argument("--memory-lines", nargs="+", required=True)
    v.add_argument("--top-k", type=int, default=3)
    v.set_defaults(func=cmd_module_vector)

    op = m_sub.add_parser(
        "openrouter-ping",
        help="Проверка OpenRouter: один chat/completions с заглушечным промптом",
    )
    op.add_argument(
        "--prompt",
        type=str,
        default="Ответь одним коротким словом: работает.",
        help="Тестовое сообщение пользователя.",
    )
    op.set_defaults(func=cmd_module_openrouter_ping)

    p = sub.add_parser("pipeline", help="Run whole pipeline")
    p_sub = p.add_subparsers(required=True)

    pj_parser = p_sub.add_parser("jsonl", help="Run pipeline over jsonl dataset")
    pj_parser.add_argument(
        "--dataset-path",
        type=str,
        default=pj_cfg.get("dataset_path"),
    )
    pj_parser.add_argument(
        "--output-path",
        type=str,
        default=pj_cfg.get("output_path"),
    )
    pj_parser.add_argument(
        "--max-user-messages",
        type=int,
        default=None,
        help="Ограничить число user-сообщений на диалог (для быстрых локальных тестов).",
    )
    pj_parser.add_argument(
        "--max-important-messages",
        type=int,
        default=None,
        help="Ограничить число важных (saved=true) сообщений на диалог (для быстрых локальных тестов).",
    )
    pj_parser.set_defaults(func=cmd_pipeline_jsonl)

    pi_parser = p_sub.add_parser("interactive", help="Run pipeline interactive mode")
    pi_parser.add_argument(
        "--dialogue-id",
        dest="dialogue_id",
        type=str,
        default=pi_cfg.get("dialogue_id", "interactive"),
    )
    pi_parser.set_defaults(func=cmd_pipeline_interactive)

    return parser


def cmd_module_classifier(args: argparse.Namespace) -> None:
    from dst_memory.classifier import ImportanceClassifier

    logger.info("Command: module classifier")
    model = ImportanceClassifier(
        model_path=args.importance_model_path, threshold=args.importance_threshold
    )
    result = model.predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_module_dst(args: argparse.Namespace) -> None:
    from dst_memory.dst_manager import DSTManager
    from dst_memory.serving import LocalHFServing
    from dst_memory.slot_client import SlotDecisionClient
    from dst_memory.slot_update_client import SlotUpdateClient

    logger.info("Command: module dst")
    slot_serving = None
    if not args.slot_use_stub:
        slot_serving = LocalHFServing(args.slot_model_path)
    slot_client = SlotDecisionClient(
        use_stub=args.slot_use_stub,
        serving=slot_serving,
        max_slots=args.slot_max_slots_per_message,
        max_retries=1,
    )
    slot_update = SlotUpdateClient(serving=slot_serving, max_retries=1)
    dst = DSTManager(slot_client=slot_client, slot_update=slot_update)
    created = dst.upsert_from_message(args.dialogue_id, args.text)
    print(json.dumps([asdict(x) for x in created], ensure_ascii=False, indent=2))


def cmd_module_openrouter_ping(args: argparse.Namespace) -> None:
    from dst_memory.llm_client import FinalLLMClient

    logger.info("Command: module openrouter-ping model=%s", args.llm_model)
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
    text = client.generate(question=args.prompt, memory_lines=[])
    print(text)


def cmd_module_vector(args: argparse.Namespace) -> None:
    from dst_memory.embedder import TextEmbedder
    from dst_memory.vector_store import InMemoryVectorStore

    logger.info("Command: module vector")
    embedder = TextEmbedder()
    store = InMemoryVectorStore()
    vectors = embedder.encode(args.memory_lines)
    for i, (line, vec) in enumerate(zip(args.memory_lines, vectors)):
        store.add(
            embedding=vec,
            payload={
                "dialogue_id": args.dialogue_id,
                "slot": "facts",
                "value": line,
                "row_id": i,
            },
        )
    q = embedder.encode([args.query])[0]
    hits = store.search(q, top_k=args.top_k)
    print(json.dumps(hits, ensure_ascii=False, indent=2))


def cmd_pipeline_jsonl(args: argparse.Namespace) -> None:
    if not args.dataset_path or not args.output_path:
        raise SystemExit(
            "pipeline jsonl requires --dataset-path and --output-path "
            "(set under pipeline_jsonl in run_config.json or pass on CLI)."
        )
    logger.info("Command: pipeline jsonl dataset_path=%s", args.dataset_path)
    pipeline = build_pipeline(args)
    rows = read_jsonl(args.dataset_path)
    results_logs = []
    results_compact = []
    for row in rows:
        dialogue_id = str(row.get("id"))
        question = row.get("question", "")
        logger.info("Processing dialogue_id=%s", dialogue_id)
        write_logs = []
        processed_messages = 0
        processed_important = 0
        for msg in iter_user_messages(row):
            if args.max_user_messages is not None and processed_messages >= args.max_user_messages:
                break
            log = pipeline.write_to_memory(dialogue_id=dialogue_id, message=msg)
            write_logs.append(log)
            processed_messages += 1
            if log.get("saved"):
                processed_important += 1
            if (
                args.max_important_messages is not None
                and processed_important >= args.max_important_messages
            ):
                break

        if args.no_final_llm:
            answer = pipeline.answer_without_final_llm(dialogue_id=dialogue_id, question=question)
        else:
            answer = pipeline.answer(dialogue_id=dialogue_id, question=question)

        results_logs.append(
            {
                "dialogue_id": dialogue_id,
                "question": question,
                "write_logs": write_logs,
                "answer": answer,
            }
        )

        compact_answer = (
            answer
            if isinstance(answer, dict)
            else pipeline.answer_without_final_llm(dialogue_id=dialogue_id, question=question)
        )
        results_compact.append(
            {
                "dialogue_id": dialogue_id,
                "question": question,
                "use_memory": compact_answer.get("use_memory"),
                "retrieved": compact_answer.get("retrieved", []),
                "memory_slots": compact_answer.get("memory_slots", []),
            }
        )
        pipeline.clear_memory(dialogue_id)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results_compact, f, ensure_ascii=False, indent=2)

    logs_output_path = args.output_path.removesuffix(".json") + "_logs.json"
    with open(logs_output_path, "w", encoding="utf-8") as f:
        json.dump(results_logs, f, ensure_ascii=False, indent=2)

    logger.info("Saved output to %s", args.output_path)
    logger.info("Saved logs output to %s", logs_output_path)
    print(f"Saved: {args.output_path}")
    print(f"Saved logs: {logs_output_path}")


def cmd_pipeline_interactive(args: argparse.Namespace) -> None:
    from dst_memory.models import Message

    pipeline = build_pipeline(args)
    did = args.dialogue_id
    logger.info("Command: pipeline interactive dialogue_id=%s", did)
    print("Interactive mode. Commands: /ask <question>, /clear, /exit")
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
        if raw.startswith("/ask "):
            q = raw[5:].strip()
            if args.no_final_llm:
                out = pipeline.answer_without_final_llm(did, q)
            else:
                out = pipeline.answer(did, q)
            print(json.dumps(out, ensure_ascii=False, indent=2) if isinstance(out, dict) else out)
            continue

        log = pipeline.write_to_memory(did, Message(role="user", content=raw))
        print(json.dumps(log, ensure_ascii=False, indent=2))


def main() -> None:
    load_dst_memory_dotenv()
    argv = sys.argv[1:]
    config_path_opt = _pre_parse_config_path(argv)
    env_path = os.environ.get("DST_MEMORY_CONFIG")
    load_path: str | Path | None = config_path_opt or env_path
    try:
        file_cfg = load_run_config(load_path)
    except FileNotFoundError as e:
        raise SystemExit(
            f"{e}\nCreate DST_memory/run_config.json or pass --config <path>."
        ) from e

    resolved = Path(load_path) if load_path else default_config_path()
    resolved = resolved.resolve()
    shared = shared_section(file_cfg)
    parser = create_parser(
        resolved_config_path=str(resolved),
        shared=shared,
        pipeline_jsonl=subsection(file_cfg, "pipeline_jsonl"),
        pipeline_interactive=subsection(file_cfg, "pipeline_interactive"),
    )
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    logger.info("DST_memory run started config=%s", resolved)
    args.func(args)


if __name__ == "__main__":
    main()
