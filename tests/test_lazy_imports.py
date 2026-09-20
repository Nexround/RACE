"""Tests that top-level and sub-package imports remain lightweight.

The core invariant is: importing a race package must not pull in heavy
optional dependencies (torch, transformers, datasets, and h5py) as a side
effect. Those should only enter ``sys.modules`` when the caller accesses a
concrete object that requires them.
"""

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEAVY_DEPS = (
    "torch",
    "transformers",
    "datasets",
    "h5py",
)


def _clean_race_modules():
    """Remove all race-related entries from sys.modules so each test starts
    from a pristine import state.  This does NOT remove already-loaded heavy
    deps; the tests only assert that *import race* alone does not introduce
    them."""
    to_remove = [
        k
        for k in sys.modules
        if k in ("race", "race_eval")
        or k.startswith("race.")
        or k.startswith("race_eval.")
    ]
    for key in to_remove:
        del sys.modules[key]


def _heavy_modules_present_before(extra_removes=()):
    """Return the set of heavy deps that are *already* in sys.modules before
    the test runs, so we can subtract them from our assertions."""
    present = {dep for dep in _HEAVY_DEPS if dep in sys.modules}
    return present


# ---------------------------------------------------------------------------
# Top-level package
# ---------------------------------------------------------------------------


class TestRaceTopLevel:
    def setup_method(self):
        _clean_race_modules()

    def test_import_race_is_lightweight(self):
        """``import race`` must not load any heavy optional dependency."""
        already_loaded = _heavy_modules_present_before()

        import race  # noqa: F401

        for dep in _HEAVY_DEPS:
            if dep in already_loaded:
                continue
            assert dep not in sys.modules, f"'import race' unexpectedly loaded {dep!r}"

    def test_race_version_accessible(self):
        import race

        assert isinstance(race.__version__, str)

    def test_dir_contains_all_exports(self):
        import race

        exported = set(race.__all__)
        listed = set(dir(race))
        assert exported <= listed, f"Missing from __dir__: {exported - listed}"

    def test_unknown_attr_raises_attribute_error(self):
        import race

        with pytest.raises(AttributeError, match="no attribute"):
            _ = race.this_does_not_exist_xyz


# ---------------------------------------------------------------------------
# race.core
# ---------------------------------------------------------------------------


class TestRaceCore:
    def setup_method(self):
        _clean_race_modules()

    def test_import_core_is_lightweight(self):
        """``import race.core`` must not load torch or h5py."""
        already_loaded = _heavy_modules_present_before()

        import race.core  # noqa: F401

        for dep in ("torch", "h5py"):
            if dep in already_loaded:
                continue
            assert (
                dep not in sys.modules
            ), f"'import race.core' unexpectedly loaded {dep!r}"

    def test_constants_immediately_available(self):
        """MODULE_ALIASES and LCB_DATASET_ALIASES are eagerly imported (no heavy
        deps) and must be accessible without attribute access tricks."""
        import race.core

        assert hasattr(race.core, "MODULE_ALIASES")
        assert hasattr(race.core, "LCB_DATASET_ALIASES")

    def test_unknown_attr_raises_attribute_error(self):
        import race.core

        with pytest.raises(AttributeError):
            _ = race.core.nonexistent_symbol_abc

    def test_removed_result_loader_exports_are_unavailable(self):
        import race.core

        for name in (
            "load_race_h5",
            "load_race_results",
            "load_race_results_with_posterior",
        ):
            assert name not in race.core.__all__
            assert name not in dir(race.core)
            with pytest.raises(AttributeError):
                getattr(race.core, name)

    def test_dir_contains_all_exports(self):
        import race.core

        exported = set(race.core.__all__)
        listed = set(dir(race.core))
        assert exported <= listed


# ---------------------------------------------------------------------------
# race_eval
# ---------------------------------------------------------------------------


class TestRaceEval:
    def setup_method(self):
        _clean_race_modules()

    def test_import_race_eval_is_lightweight(self):
        already_loaded = _heavy_modules_present_before()

        import race_eval  # noqa: F401

        for dep in _HEAVY_DEPS:
            if dep in already_loaded:
                continue
            assert (
                dep not in sys.modules
            ), f"'import race_eval' unexpectedly loaded {dep!r}"

    def test_ablation_plan_lazy(self):
        already_loaded = _heavy_modules_present_before()

        import race_eval.ablation

        _ = race_eval.ablation.AblationPlan
        assert "race_eval.ablation.plan" in sys.modules
        if "h5py" not in already_loaded:
            assert "h5py" not in sys.modules

    def test_dir_contains_all_exports(self):
        import race_eval.ablation

        exported = set(race_eval.ablation.__all__)
        listed = set(dir(race_eval.ablation))
        assert exported <= listed

    def test_unknown_attr_raises_attribute_error(self):
        import race_eval

        with pytest.raises(AttributeError):
            _ = race_eval.no_such_thing_xyz


# ---------------------------------------------------------------------------
# race.llm
# ---------------------------------------------------------------------------


class TestRaceLlm:
    def setup_method(self):
        _clean_race_modules()

    def test_import_llm_is_lightweight(self):
        already_loaded = _heavy_modules_present_before()

        import race.llm  # noqa: F401

        for dep in ("torch", "transformers"):
            if dep in already_loaded:
                continue
            assert (
                dep not in sys.modules
            ), f"'import race.llm' unexpectedly loaded {dep!r}"

    def test_unknown_attr_raises_attribute_error(self):
        import race.llm

        with pytest.raises(AttributeError):
            _ = race.llm.no_such_thing_xyz

    def test_dir_contains_all_exports(self):
        import race.llm

        exported = set(race.llm.__all__)
        listed = set(dir(race.llm))
        assert exported <= listed


# ---------------------------------------------------------------------------
# race.io
# ---------------------------------------------------------------------------


class TestRaceIo:
    def setup_method(self):
        _clean_race_modules()

    def test_import_io_is_lightweight(self):
        already_loaded = _heavy_modules_present_before()

        import race.io  # noqa: F401

        if "h5py" not in already_loaded:
            assert (
                "h5py" not in sys.modules
            ), "'import race.io' unexpectedly loaded h5py"

    def test_unknown_attr_raises_attribute_error(self):
        import race.io

        with pytest.raises(AttributeError):
            _ = race.io.nonexistent_xyz


# ---------------------------------------------------------------------------
# race.utils
# ---------------------------------------------------------------------------


class TestRaceUtils:
    def setup_method(self):
        _clean_race_modules()

    def test_import_utils_is_lightweight(self):
        already_loaded = _heavy_modules_present_before()

        import race.utils  # noqa: F401

        if "torch" not in already_loaded:
            assert (
                "torch" not in sys.modules
            ), "'import race.utils' unexpectedly loaded torch"

    def test_unknown_attr_raises_attribute_error(self):
        import race.utils

        with pytest.raises(AttributeError):
            _ = race.utils.nonexistent_xyz


# ---------------------------------------------------------------------------
# race._lazy helper unit tests
# ---------------------------------------------------------------------------


class TestLazyHelper:
    """Unit-test the _lazy.load_lazy_attr helper directly."""

    def test_resolves_attribute(self):
        from race._lazy import load_lazy_attr

        _globals: dict = {}
        result = load_lazy_attr(
            _globals,
            "race_eval.ablation",
            {"AblationPlan": (".plan", "AblationPlan")},
            "AblationPlan",
        )
        from race_eval.ablation import AblationPlan

        assert result is AblationPlan

    def test_caches_in_globals(self):
        from race._lazy import load_lazy_attr

        _globals: dict = {}
        load_lazy_attr(
            _globals,
            "race_eval.ablation",
            {"AblationPlan": (".plan", "AblationPlan")},
            "AblationPlan",
        )
        assert "AblationPlan" in _globals

    def test_missing_name_raises_attribute_error(self):
        from race._lazy import load_lazy_attr

        with pytest.raises(AttributeError, match="no attribute"):
            load_lazy_attr({}, "race_eval.ablation", {}, "does_not_exist")
