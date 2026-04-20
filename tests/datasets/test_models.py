from __future__ import annotations

from layerlens.datasets.models import (
    Dataset,
    DatasetItem,
    DatasetVersion,
    DatasetVisibility,
)


class TestDatasetItem:
    def test_defaults(self):
        item = DatasetItem(id="i1", input="x")
        assert item.expected_output is None
        assert item.metadata == {}
        assert item.tags == []


class TestDatasetVersion:
    def test_size(self):
        v = DatasetVersion(version=1, items=[DatasetItem(id="a", input=1)])
        assert v.size == 1

    def test_version_must_be_positive(self):
        import pydantic

        try:
            DatasetVersion(version=0)
        except pydantic.ValidationError:
            return
        raise AssertionError("expected ValidationError for version=0")


class TestDatasetHelpers:
    def test_latest_returns_highest_version(self):
        ds = Dataset(
            id="d",
            name="n",
            versions=[DatasetVersion(version=2), DatasetVersion(version=5)],
        )
        latest = ds.latest()
        assert latest is not None
        assert latest.version == 5

    def test_latest_with_no_versions_is_none(self):
        ds = Dataset(id="d", name="n")
        assert ds.latest() is None

    def test_lookup_by_version(self):
        ds = Dataset(
            id="d",
            name="n",
            versions=[DatasetVersion(version=1), DatasetVersion(version=2)],
        )
        assert ds.version(2).version == 2
        assert ds.version(9) is None

    def test_visibility_enum(self):
        ds = Dataset(id="d", name="n", visibility=DatasetVisibility.PUBLIC)
        assert ds.visibility == DatasetVisibility.PUBLIC
