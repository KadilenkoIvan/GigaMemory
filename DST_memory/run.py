import argparse
import json
import logging
from dataclasses import asdict

from dst_memory.classifier import ImportanceClassifier
from dst_memory.dst_manager import DSTManager
from dst_memory.io_utils import iter_user_messages, read_jsonl


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


logger = logging.getLogger(__name__)


def build_pipeline(args: argparse.Namespace):
    from dst_memory import PipelineConfig
    from dst_memory.pipeline import DSTMemoryPipeline

    cfg = PipelineConfig(
        importance_model_path=args.importance_model_path,
        importance_threshold=args.importance_threshold,
        retrieval_top_k=args.retrieval_top_k,
        use_memory_gate=not args.disable_memory_gate,
        llm_mode=args.llm_mode,
        llm_api_url=args.llm_api_url,
        llm_api_key=args.llm_api_key,
        slot_use_stub=args.slot_use_stub,
        slot_model_path=args.slot_model_path,
        slot_max_slots_per_message=args.slot_max_slots_per_message,
    )
    return DSTMemoryPipeline(cfg)


def cmd_module_classifier(args: argparse.Namespace) -> None:
    logger.info("Command: module classifier")
    model = ImportanceClassifier(
        model_path=args.importance_model_path, threshold=args.importance_threshold
    )
    result = model.predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_module_dst(args: argparse.Namespace) -> None:
    logger.info("Command: module dst")
    from dst_memory.slot_client import SlotDecisionClient

    slot_client = SlotDecisionClient(
        use_stub=args.slot_use_stub,
        model_path=args.slot_model_path,
        max_slots=args.slot_max_slots_per_message,
        max_retries=1,
    )
    dst = DSTManager(slot_client=slot_client)
    created = dst.upsert_from_message(args.dialogue_id, args.text)
    print(json.dumps([asdict(x) for x in created], ensure_ascii=False, indent=2))


def cmd_module_vector(args: argparse.Namespace) -> None:
    logger.info("Command: module vector")
    from dst_memory.embedder import TextEmbedder
    from dst_memory.vector_store import InMemoryVectorStore

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
        for msg in iter_user_messages(row):
            log = pipeline.write_to_memory(dialogue_id=dialogue_id, message=msg)
            write_logs.append(log)

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

        # Compact output: only what is currently stored in memory state.
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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DST_memory runner")
    parser.add_argument(
        "--importance-model-path",
        type=str,
        default="message_important_learning/best_model-full_tune",
    )
    parser.add_argument("--importance-threshold", type=float, default=0.5)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--disable-memory-gate", action="store_true")
    parser.add_argument("--llm-mode", type=str, default="stub", choices=["stub", "local", "api"])
    parser.add_argument("--llm-api-url", type=str, default="")
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--no-final-llm", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--slot-use-stub", action="store_true")
    parser.add_argument("--slot-model-path", type=str, default="models/Meno-Lite-0.1")
    parser.add_argument("--slot-max-slots-per-message", type=int, default=5)

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

    p = sub.add_parser("pipeline", help="Run whole pipeline")
    p_sub = p.add_subparsers(required=True)

    pj = p_sub.add_parser("jsonl", help="Run pipeline over jsonl dataset")
    pj.add_argument("--dataset-path", required=True, type=str)
    pj.add_argument("--output-path", required=True, type=str)
    pj.set_defaults(func=cmd_pipeline_jsonl)

    pi = p_sub.add_parser("interactive", help="Run pipeline interactive mode")
    pi.add_argument("--dialogue-id", default="interactive", type=str)
    pi.set_defaults(func=cmd_pipeline_interactive)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    logger.info("DST_memory run started")
    args.func(args)


if __name__ == "__main__":
    main()
