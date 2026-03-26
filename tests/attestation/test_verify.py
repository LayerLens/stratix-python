from __future__ import annotations

from layerlens.attestation import (
    HashChain,
    HashScope,
    verify_chain,
    verify_trial,
    detect_tampering,
)


class TestVerifyChain:
    def test_valid_chain(self):
        chain = HashChain()
        chain.add_event({"name": "a"})
        chain.add_event({"name": "b"})
        chain.add_event({"name": "c"})
        result = verify_chain(chain.envelopes)
        assert result.valid
        assert result.break_index is None

    def test_empty_chain_valid(self):
        result = verify_chain([])
        assert result.valid

    def test_single_event_valid(self):
        chain = HashChain()
        chain.add_event({"name": "a"})
        result = verify_chain(chain.envelopes)
        assert result.valid

    def test_broken_first_link(self):
        """First envelope must have previous_hash=None."""
        chain = HashChain()
        chain.add_event({"name": "a"})
        envelopes = chain.envelopes
        # Tamper: set previous_hash on first event
        envelopes[0].previous_hash = "sha256:fake"
        result = verify_chain(envelopes)
        assert not result.valid
        assert result.break_index == 0

    def test_broken_middle_link(self):
        chain = HashChain()
        chain.add_event({"name": "a"})
        chain.add_event({"name": "b"})
        chain.add_event({"name": "c"})
        envelopes = chain.envelopes
        # Tamper: break the link between event 1 and 2
        envelopes[2].previous_hash = "sha256:fake"
        result = verify_chain(envelopes)
        assert not result.valid
        assert result.break_index == 2


class TestVerifyTrial:
    def test_valid_trial(self):
        chain = HashChain()
        chain.add_event({"name": "a"})
        chain.add_event({"name": "b"})
        envelopes = chain.envelopes
        trial = chain.finalize()
        result = verify_trial(envelopes, trial)
        assert result.valid

    def test_wrong_scope_rejected(self):
        chain = HashChain()
        chain.add_event({"name": "a"})
        envelopes = chain.envelopes
        trial = chain.finalize()
        trial.scope = HashScope.EVENT  # Wrong scope
        result = verify_trial(envelopes, trial)
        assert not result.valid
        assert not result.trial_hash_valid
        assert any("scope" in e for e in result.errors)

    def test_tampered_trial_hash(self):
        chain = HashChain()
        chain.add_event({"name": "a"})
        envelopes = chain.envelopes
        trial = chain.finalize()
        trial.hash = "sha256:" + "0" * 64  # Wrong hash
        result = verify_trial(envelopes, trial)
        assert not result.valid
        assert not result.trial_hash_valid
        assert any("does not match" in e for e in result.errors)


class TestDetectTampering:
    def test_no_tampering(self):
        data = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        chain = HashChain()
        for d in data:
            chain.add_event(d)
        result = detect_tampering(chain.envelopes, data)
        assert not result.tampered
        assert result.modified_indices == []
        assert not result.chain_broken

    def test_detect_modified_event(self):
        data = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        chain = HashChain()
        for d in data:
            chain.add_event(d)
        # Tamper with the second event's data
        tampered_data = [{"name": "a"}, {"name": "CHANGED"}, {"name": "c"}]
        result = detect_tampering(chain.envelopes, tampered_data)
        assert result.tampered
        assert 1 in result.modified_indices

    def test_detect_multiple_modifications(self):
        data = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        chain = HashChain()
        for d in data:
            chain.add_event(d)
        tampered = [{"name": "X"}, {"name": "b"}, {"name": "Z"}]
        result = detect_tampering(chain.envelopes, tampered)
        assert result.tampered
        assert 0 in result.modified_indices
        assert 2 in result.modified_indices

    def test_detect_count_mismatch(self):
        data = [{"name": "a"}, {"name": "b"}]
        chain = HashChain()
        for d in data:
            chain.add_event(d)
        result = detect_tampering(chain.envelopes, [{"name": "a"}])
        assert result.tampered
        assert result.chain_broken

    def test_detect_tampering_with_signed_chain(self):
        """detect_tampering works correctly on chains that were signed."""
        data = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        chain = HashChain(signing_key_id="org-1", signing_secret=b"test-key")
        for d in data:
            chain.add_event(d)

        # No tampering — should pass
        result = detect_tampering(chain.envelopes, data)
        assert not result.tampered
        assert result.modified_indices == []

        # Tamper with one event
        tampered = [{"name": "a"}, {"name": "CHANGED"}, {"name": "c"}]
        result = detect_tampering(chain.envelopes, tampered)
        assert result.tampered
        assert 1 in result.modified_indices
