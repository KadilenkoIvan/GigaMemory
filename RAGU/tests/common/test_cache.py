from pathlib import Path

from ragu.common.cache import _CACHES, get_cache

# TODO: add more tests for new version of cache
def test_get_cache_returns_same_instance_for_same_path(tmp_path):
    _CACHES.clear()
    cache_dir = tmp_path / "cache_a"

    first = get_cache(cache_dir)
    second = get_cache(cache_dir)

    assert first is second


def test_get_cache_returns_different_instances_for_different_paths(tmp_path):
    _CACHES.clear()
    first = get_cache(tmp_path / "cache_a")
    second = get_cache(tmp_path / "cache_b")

    assert first is not second


def test_cache_persists_values_via_diskcache_index(tmp_path):
    _CACHES.clear()
    cache_dir = Path(tmp_path) / "cache_persist"
    cache = get_cache(cache_dir)
    cache["k"] = {"value": 123}

    reloaded = get_cache(cache_dir)
    assert reloaded["k"] == {"value": 123}
