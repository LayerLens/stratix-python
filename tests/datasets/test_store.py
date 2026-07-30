from __future__ import annotations

import pytest

from layerlens.datasets.store import InMemoryDatasetStore
from layerlens.datasets.models import Dataset, DatasetItem, DatasetVisibility


@pytest.fixture
def store():
    return InMemoryDatasetStore()


class TestCreate:
    def test_assigns_id_when_empty(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        assert ds.id.startswith("ds_")

    def test_seeds_initial_version(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        assert ds.current_version == 1
        assert ds.latest() is not None

    def test_duplicate_id_raises(self, store):
        ds = store.create(Dataset(id="explicit", name="qa"))
        with pytest.raises(ValueError):
            store.create(Dataset(id=ds.id, name="other"))


class TestPublishVersion:
    def test_appends_and_bumps_current(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        v = store.publish_version(ds.id, [DatasetItem(id="i1", input=1)], note="seed")
        assert v is not None
        assert v.version == 2
        assert ds.current_version == 2
        assert v.size == 1

    def test_publish_missing_dataset_returns_none(self, store):
        assert store.publish_version("missing", []) is None


class TestMetadata:
    def test_update_name_description_tags(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        updated = store.update_metadata(ds.id, name="renamed", description="desc", tags=["rag", "eval"])
        assert updated is not None
        assert updated.name == "renamed"
        assert updated.description == "desc"
        assert updated.tags == ["rag", "eval"]
        assert updated.updated_at >= ds.created_at

    def test_update_missing_returns_none(self, store):
        assert store.update_metadata("missing", name="x") is None


class TestDelete:
    def test_delete_returns_true_when_present(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        assert store.delete(ds.id) is True
        assert store.get(ds.id) is None

    def test_delete_missing_returns_false(self, store):
        assert store.delete("missing") is False


class TestList:
    def test_filter_by_tag(self, store):
        a = store.create(Dataset(id="", name="a", tags=["rag"]))
        store.create(Dataset(id="", name="b", tags=["tooling"]))
        matched = store.list(tag="rag")
        assert [d.id for d in matched] == [a.id]

    def test_filter_by_org_and_project(self, store):
        a = store.create(Dataset(id="", name="a", organization_id="o1", project_id="p1"))
        store.create(Dataset(id="", name="b", organization_id="o2", project_id="p1"))
        assert [d.id for d in store.list(organization_id="o1")] == [a.id]
        assert len(store.list(project_id="p1")) == 2

    def test_filter_by_visibility(self, store):
        a = store.create(Dataset(id="", name="a", visibility=DatasetVisibility.PUBLIC))
        store.create(Dataset(id="", name="b", visibility=DatasetVisibility.PRIVATE))
        assert [d.id for d in store.list(visibility=DatasetVisibility.PUBLIC)] == [a.id]

    def test_sorted_by_updated_at_desc(self, store):
        a = store.create(Dataset(id="", name="a"))
        b = store.create(Dataset(id="", name="b"))
        # Bump a so it's newer.
        store.update_metadata(a.id, description="touched")
        ordered = [d.id for d in store.list()]
        assert ordered[0] == a.id
        assert set(ordered) == {a.id, b.id}


class TestIterItems:
    def test_defaults_to_latest_version(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        store.import_items(ds.id, [{"input": 1}, {"input": 2}])
        store.import_items(ds.id, [{"input": 3}])
        assert [i.input for i in store.iter_items(ds.id)] == [3]

    def test_specific_version(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        store.import_items(ds.id, [{"input": "a"}])
        store.import_items(ds.id, [{"input": "b"}])
        v1 = [i.input for i in store.iter_items(ds.id, version=2)]
        assert v1 == ["a"]

    def test_filter_by_tag(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        store.import_items(
            ds.id,
            [
                {"input": 1, "tags": ["smoke"]},
                {"input": 2, "tags": ["regression"]},
            ],
        )
        smoke = list(store.iter_items(ds.id, tag="smoke"))
        assert len(smoke) == 1 and smoke[0].input == 1

    def test_missing_dataset_returns_empty(self, store):
        assert list(store.iter_items("missing")) == []


class TestImportItems:
    def test_generates_ids_if_missing(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        v = store.import_items(ds.id, [{"input": 1}, {"input": 2}])
        assert v.size == 2
        assert all(item.id for item in v.items)

    def test_preserves_supplied_id(self, store):
        ds = store.create(Dataset(id="", name="qa"))
        v = store.import_items(ds.id, [{"id": "keep-me", "input": 1}])
        assert v.items[0].id == "keep-me"
