"""Unit tests for race.io.h5_schema — HDF5 schema utilities."""

import json
import tempfile

import h5py
import numpy as np
import pytest

from race.io.h5_schema import (
    SCHEMA_VERSION,
    ensure_meta,
    ensure_instances_group,
    ensure_axis_value,
    write_json_blob,
    format_instance_key,
    update_total_instances,
)


@pytest.fixture
def tmp_h5(tmp_path):
    """Create a temporary H5 file for testing."""
    path = str(tmp_path / "test.h5")
    yield path


class TestEnsureMeta:
    def test_creates_meta_group(self, tmp_h5):
        with h5py.File(tmp_h5, "w") as f:
            meta_group, axes_group = ensure_meta(
                f,
                target_model_family="vit",
                model_name="test-model",
                device="cpu",
            )
            assert "meta" in f
            assert "run" in f["meta"]
            assert "schema" in f["meta"]
            assert f["meta"]["schema"].attrs["version"] == SCHEMA_VERSION

    def test_idempotent(self, tmp_h5):
        with h5py.File(tmp_h5, "w") as f:
            ensure_meta(f, target_model_family="vit", model_name="m", device="cpu")
            ensure_meta(f, target_model_family="vit", model_name="m", device="cpu")
            # Should not raise


class TestEnsureInstancesGroup:
    def test_creates_instances(self, tmp_h5):
        with h5py.File(tmp_h5, "w") as f:
            grp = ensure_instances_group(f)
            assert "instances" in f
            assert isinstance(grp, h5py.Group)


class TestFormatInstanceKey:
    def test_formatting(self):
        assert format_instance_key(0) == "instance_000000"
        assert format_instance_key(42) == "instance_000042"
        assert format_instance_key(123456) == "instance_123456"


class TestUpdateTotalInstances:
    def test_basic(self, tmp_h5):
        with h5py.File(tmp_h5, "w") as f:
            ensure_meta(f, target_model_family="vit", model_name="m", device="cpu")
            update_total_instances(f, 100)
            assert f["meta"]["run"].attrs["total_instances"] == 100


class TestWriteJsonBlob:
    def test_basic(self, tmp_h5):
        with h5py.File(tmp_h5, "w") as f:
            grp = f.create_group("test")
            payload = {"key": "value", "number": 42}
            write_json_blob(grp, "payload", payload)
            # Read back — stored as a 1-element string array
            raw = grp["payload"][0]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(str(raw))
            assert data["key"] == "value"
            assert data["number"] == 42


class TestEnsureAxisValue:
    def test_basic(self, tmp_h5):
        with h5py.File(tmp_h5, "w") as f:
            _, axes_group = ensure_meta(
                f, target_model_family="vit", model_name="m", device="cpu"
            )
            value_group, path = ensure_axis_value(
                axes_group,
                axis_type="predicted_label",
                value_key=42,
                display_name="cat",
                raw_value={"label": 42},
            )
            assert isinstance(value_group, h5py.Group)
            assert isinstance(path, str)
            assert value_group.attrs["display_name"] == "cat"
