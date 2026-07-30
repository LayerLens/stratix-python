from __future__ import annotations

import pytest

from layerlens.attestation._chain import HashChain
from layerlens.attestation._envelope import HashScope


class TestHashChainBuilding:
    def test_single_event(self):
        chain = HashChain()
        env = chain.add_event({"name": "span-1"})
        assert env.previous_hash is None
        assert env.scope == HashScope.EVENT
        assert env.hash.startswith("sha256:")

    def test_chain_linking(self):
        """Each event links to the previous hash."""
        chain = HashChain()
        e1 = chain.add_event({"name": "span-1"})
        e2 = chain.add_event({"name": "span-2"})
        e3 = chain.add_event({"name": "span-3"})

        assert e1.previous_hash is None
        assert e2.previous_hash == e1.hash
        assert e3.previous_hash == e2.hash

    def test_different_data_different_hashes(self):
        chain = HashChain()
        e1 = chain.add_event({"name": "a"})
        e2 = chain.add_event({"name": "b"})
        assert e1.hash != e2.hash

    def test_envelopes_property(self):
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        chain.add_event({"name": "span-2"})
        assert len(chain.envelopes) == 2


class TestHashChainFinalization:
    def test_finalize_produces_trial_scope(self):
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        trial = chain.finalize()
        assert trial.scope == HashScope.TRIAL

    def test_finalize_root_hash_deterministic(self):
        """Same events in same order produce the same root hash."""

        def build():
            c = HashChain()
            c.add_event({"name": "a"})
            c.add_event({"name": "b"})
            return c.finalize()

        assert build().hash == build().hash

    def test_finalize_seals_chain(self):
        """No events can be added after finalization."""
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        chain.finalize()
        with pytest.raises(RuntimeError, match="terminated"):
            chain.add_event({"name": "span-2"})

    def test_finalize_empty_chain_raises(self):
        chain = HashChain()
        with pytest.raises(RuntimeError, match="empty"):
            chain.finalize()

    def test_finalize_links_to_last_event(self):
        chain = HashChain()
        chain.add_event({"name": "a"})
        last = chain.add_event({"name": "b"})
        trial = chain.finalize()
        assert trial.previous_hash == last.hash


class TestHashChainTermination:
    def test_terminate_blocks_add(self):
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        chain.terminate("policy_violation")
        with pytest.raises(RuntimeError, match="terminated"):
            chain.add_event({"name": "span-2"})

    def test_terminate_blocks_finalize(self):
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        chain.terminate("policy_violation")
        with pytest.raises(RuntimeError, match="non-attestable"):
            chain.finalize()

    def test_is_terminated_flag(self):
        chain = HashChain()
        assert not chain.is_terminated
        chain.terminate("test")
        assert chain.is_terminated

    def test_terminate_reason_in_error(self):
        chain = HashChain()
        chain.terminate("safety_check_failed")
        with pytest.raises(RuntimeError, match="safety_check_failed"):
            chain.add_event({"name": "span-1"})


class TestHashChainSerialization:
    def test_to_dict(self):
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        d = chain.to_dict()
        assert "events" in d
        assert len(d["events"]) == 1
        assert d["events"][0]["scope"] == "event"
        assert d["events"][0]["hash"].startswith("sha256:")

    def test_to_dict_finalized_is_clean(self):
        """Normal finalization should not include termination details."""
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        chain.finalize()
        d = chain.to_dict()
        assert "terminated_reason" not in d

    def test_to_dict_terminated_includes_reason(self):
        """Policy violation termination should include the reason."""
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        chain.terminate("policy_violation")
        d = chain.to_dict()
        assert d["terminated_reason"] == "policy_violation"
