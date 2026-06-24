"""
Tests for real-time parallel write mode and session persistence.

All tests run in stub mode — no GPU, no real LLM required.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dst_memory.core.config import PipelineConfig
from dst_memory.core.models import DialogueMemoryState, FactRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_config(**overrides) -> PipelineConfig:
    base = dict(
        slot_use_stub=True,
        llm_mode="stub",
        use_ragu=True,
        ragu_storage_path="",
        importance_model_path="",
        slot_model_path="",
        ragu_embedder_model="",
    )
    base.update(overrides)
    return PipelineConfig(**base)


def _make_mock_ragu(tmp_path: Path):
    """Return a minimal RAGU processor mock that satisfies DSTMemoryPipeline."""
    mock = MagicMock()
    mock.upsert_triplet_deltas.return_value = None
    mock.delete_triplet_deltas.return_value = None
    mock.search_memory.return_value = []
    mock.clear_all.return_value = None
    return mock


def _make_pipeline(tmp_path: Path, **cfg_overrides):
    """Build a stub pipeline with a mock RAGU processor (no GPU/models needed)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RAGU"))
    from dst_memory.core.pipeline import DSTMemoryPipeline

    cfg = _make_stub_config(
        ragu_storage_path=str(tmp_path / "ragu"),
        **cfg_overrides,
    )
    ragu = _make_mock_ragu(tmp_path)
    return DSTMemoryPipeline(cfg, ragu_processor=ragu)


# ---------------------------------------------------------------------------
# PipelineConfig defaults
# ---------------------------------------------------------------------------


class TestParallelWriteConfig:
    def test_parallel_write_mode_default_false(self) -> None:
        cfg = PipelineConfig()
        assert cfg.parallel_write_mode is False

    def test_parallel_write_mode_can_be_set(self) -> None:
        cfg = PipelineConfig(parallel_write_mode=True)
        assert cfg.parallel_write_mode is True

    def test_session_dir_default_empty(self) -> None:
        cfg = PipelineConfig()
        assert cfg.session_dir == ""

    def test_session_dir_can_be_set(self) -> None:
        cfg = PipelineConfig(session_dir="/tmp/sessions")
        assert cfg.session_dir == "/tmp/sessions"


# ---------------------------------------------------------------------------
# FinalLLMClient parallel_write_mode
# ---------------------------------------------------------------------------


class TestFinalLLMClientParallelMode:
    def test_parallel_write_mode_false_by_default(self) -> None:
        from dst_memory.clients.llm_client import FinalLLMClient

        client = FinalLLMClient(mode="stub")
        assert client.parallel_write_mode is False

    def test_parallel_write_mode_stored(self) -> None:
        from dst_memory.clients.llm_client import FinalLLMClient

        client = FinalLLMClient(mode="stub", parallel_write_mode=True)
        assert client.parallel_write_mode is True

    def test_no_notice_when_disabled(self) -> None:
        from dst_memory.clients.llm_client import FinalLLMClient

        client = FinalLLMClient(
            mode="stub", parallel_write_mode=False, prompt_language="en"
        )
        msgs = client.build_messages("hello", {})
        system = msgs[0]["content"]
        assert "parallel" not in system.lower()

    def test_notice_injected_when_enabled_en(self) -> None:
        from dst_memory.clients.llm_client import FinalLLMClient

        client = FinalLLMClient(
            mode="stub", parallel_write_mode=True, prompt_language="en"
        )
        msgs = client.build_messages("hello", {})
        system = msgs[0]["content"]
        assert "parallel" in system.lower()
        assert "recent conversation pairs" in system.lower()

    def test_notice_injected_when_enabled_ru(self) -> None:
        from dst_memory.clients.llm_client import FinalLLMClient

        client = FinalLLMClient(
            mode="stub", parallel_write_mode=True, prompt_language="ru"
        )
        msgs = client.build_messages("hello", {})
        system = msgs[0]["content"]
        assert "параллельн" in system.lower()


# ---------------------------------------------------------------------------
# Prompt modules have parallel_write_notice()
# ---------------------------------------------------------------------------


class TestPromptModules:
    def test_en_has_parallel_write_notice(self) -> None:
        import importlib

        pm = importlib.import_module("dst_memory.prompts.en.final_llm_messages")
        assert hasattr(pm, "parallel_write_notice")
        notice = pm.parallel_write_notice()
        assert isinstance(notice, str) and len(notice) > 10

    def test_ru_has_parallel_write_notice(self) -> None:
        import importlib

        pm = importlib.import_module("dst_memory.prompts.ru.final_llm_messages")
        assert hasattr(pm, "parallel_write_notice")
        notice = pm.parallel_write_notice()
        assert isinstance(notice, str) and len(notice) > 10

    def test_ru_notice_is_russian(self) -> None:
        import importlib

        pm = importlib.import_module("dst_memory.prompts.ru.final_llm_messages")
        notice = pm.parallel_write_notice()
        assert any(ord(c) > 1000 for c in notice), "Expected Cyrillic characters"


# ---------------------------------------------------------------------------
# Session save / load
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        did = "test_dialogue"
        # Create some state
        state = pipeline.dst.get_state(did)
        state.step = 3
        state.recent_pairs = [{"role": "user", "content": "hi"}]

        session_dir = str(tmp_path / "sessions")
        pipeline.save_session(did, session_dir)

        saved = Path(session_dir) / did / "state.json"
        assert saved.exists()

    def test_saved_content_is_valid_json(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        did = "test_dialogue"
        state = pipeline.dst.get_state(did)
        state.step = 5

        session_dir = str(tmp_path / "sessions")
        pipeline.save_session(did, session_dir)

        saved = Path(session_dir) / did / "state.json"
        data = json.loads(saved.read_text(encoding="utf-8"))
        assert data["dialogue_id"] == did
        assert data["step"] == 5

    def test_load_restores_step(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        did = "test_dialogue"
        state = pipeline.dst.get_state(did)
        state.step = 7
        state.recent_pairs = [{"role": "user", "content": "hello"}]

        session_dir = str(tmp_path / "sessions")
        pipeline.save_session(did, session_dir)

        # Create fresh pipeline and load
        pipeline2 = _make_pipeline(tmp_path)
        result = pipeline2.load_session(did, session_dir)

        assert result is True
        loaded = pipeline2.dst.get_state(did)
        assert loaded.step == 7
        assert loaded.recent_pairs == [{"role": "user", "content": "hello"}]

    def test_load_returns_false_if_no_file(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.load_session("nonexistent", str(tmp_path / "sessions"))
        assert result is False

    def test_save_load_roundtrip_with_facts(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        did = "test_dialogue"
        state = pipeline.dst.get_state(did)
        state.step = 2
        state.next_record_id = 3
        fact = FactRecord(
            record_id=1,
            value="пользователь | работает в | Яндекс",
            source_text="Я работаю в Яндексе",
            created_at_step=1,
            updated_at_step=1,
            subject="пользователь",
            relation="работает в",
            object="Яндекс",
            is_active=True,
            ttl="inf",
        )
        state.slots["WORK"] = [fact]

        session_dir = str(tmp_path / "sessions")
        pipeline.save_session(did, session_dir)

        pipeline2 = _make_pipeline(tmp_path)
        pipeline2.load_session(did, session_dir)
        loaded = pipeline2.dst.get_state(did)

        assert "WORK" in loaded.slots
        assert len(loaded.slots["WORK"]) == 1
        r = loaded.slots["WORK"][0]
        assert r.subject == "пользователь"
        assert r.relation == "работает в"
        assert r.object == "Яндекс"
        assert r.ttl == "inf"

    def test_save_is_idempotent(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        did = "test_dialogue"
        pipeline.dst.get_state(did).step = 1
        session_dir = str(tmp_path / "sessions")

        pipeline.save_session(did, session_dir)
        pipeline.dst.get_state(did).step = 2
        pipeline.save_session(did, session_dir)

        pipeline2 = _make_pipeline(tmp_path)
        pipeline2.load_session(did, session_dir)
        assert pipeline2.dst.get_state(did).step == 2


# ---------------------------------------------------------------------------
# Thread safety: parallel write does not block answer
# ---------------------------------------------------------------------------


class TestParallelWriteThreading:
    def test_write_thread_completes(self, tmp_path: Path) -> None:
        """Background write thread must complete without exception."""
        results: list = []
        errors: list = []

        def _write():
            try:
                results.append("done")
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=_write, daemon=True)
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert results == ["done"]
        assert not errors

    def test_parallel_write_and_answer_independent(self, tmp_path: Path) -> None:
        """write_to_memory and answer() can run on separate threads without deadlock."""
        import time

        pipeline = _make_pipeline(tmp_path)
        did = "thread_test"

        write_done = threading.Event()
        answer_done = threading.Event()

        def _write():
            from dst_memory.core.models import Message

            pipeline.write_to_memory(
                did, Message(role="user", content="я живу в Москве")
            )
            write_done.set()

        def _answer():
            pipeline.answer(did, "где я живу?")
            answer_done.set()

        t1 = threading.Thread(target=_write, daemon=True)
        t2 = threading.Thread(target=_answer, daemon=True)
        t1.start()
        t2.start()

        assert write_done.wait(timeout=30), "write_to_memory timed out"
        assert answer_done.wait(timeout=30), "answer() timed out"
