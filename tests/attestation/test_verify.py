from __future__ import annotations

from layerlens.attestation import (
    HashChain,
    HashScope,
    verify_chain,
    verify_trial,
    detect_tampering,
)
from layerlens.attestation._hash import compute_hash
from layerlens.attestation._envelope import AttestationEnvelope


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

    def test_detect_tampering_with_multi_event_chain(self):
        """detect_tampering works correctly on multi-event chains."""
        data = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        chain = HashChain()
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


# ---------------------------------------------------------------------------
# End-to-end tamper detection for the FULL attack surface (LAY-3572 / integrity).
# The existing tests cover modify (detect_tampering). These add REORDER, DELETE,
# and (the subtle one) a FULLY-RELINKED INSERT — which keeps the event chain
# internally consistent, so detect_tampering alone returns tampered=False. The
# finalized TRIAL ROOT (computed over the original event hashes, server-anchored
# at ingest) is what catches it. These are the population-complete tamper
# invariants the audit found missing.
# ---------------------------------------------------------------------------


class TestTamperDetectionEndToEnd:
    def _chain(self, n=3):
        data = [{"event": i, "name": chr(ord("a") + i)} for i in range(n)]
        chain = HashChain()
        for d in data:
            chain.add_event(d)
        return chain, data

    def test_reorder_is_detected(self):
        chain, data = self._chain(3)
        envs = chain.envelopes
        # swap events 1 and 2 (both envelope and data, as an attacker would)
        envs[1], envs[2] = envs[2], envs[1]
        data[1], data[2] = data[2], data[1]
        result = detect_tampering(envs, data)
        assert result.tampered and result.chain_broken, "event reorder not detected"

    def test_delete_of_event_pair_is_detected(self):
        chain, data = self._chain(3)
        envs = chain.envelopes
        # drop the middle (event, envelope) pair — counts still match, but the
        # surviving link no longer chains.
        del envs[1]
        del data[1]
        result = detect_tampering(envs, data)
        assert result.tampered and result.chain_broken, "event deletion not detected"

    def test_fully_relinked_insert_is_caught_by_trial_root(self):
        chain, data = self._chain(3)
        trial = chain.finalize()  # server-anchored root over the ORIGINAL 3 hashes
        envs = chain.envelopes
        # Forge an event between index 0 and 1, recomputing its hash AND relinking
        # the following envelope so the CHAIN stays internally consistent.
        forged_data = {"event": 99, "name": "FORGED"}
        forged_hash = compute_hash({**forged_data, "_previous_hash": envs[0].hash})
        forged_env = AttestationEnvelope(hash=forged_hash, scope=HashScope.EVENT, previous_hash=envs[0].hash)
        envs[1].previous_hash = forged_hash  # relink the real event 1 onto the forgery
        tampered_envs = [envs[0], forged_env, envs[1], envs[2]]

        # The chain itself now verifies clean — proving detect_tampering/chain
        # checks are NOT sufficient against a relinked insert:
        assert verify_chain(tampered_envs).valid, "precondition: relinked chain should look continuous"

        # ...but the finalized trial root (over the original event-hash set) does
        # not match the tampered event set -> the insert IS caught.
        result = verify_trial(tampered_envs, trial)
        assert not result.valid, "relinked insert slipped past the trial-root check"
        assert not result.trial_hash_valid, "trial root failed to detect the inserted event"

    def test_verify_trial_reports_broken_event_chain(self):
        chain, _ = self._chain(3)
        envs = chain.envelopes
        trial = chain.finalize()
        envs[1].previous_hash = "sha256:" + "0" * 64  # break the link
        result = verify_trial(envs, trial)
        assert not result.valid
        assert not result.chain_valid
        assert any("Chain integrity failed" in e for e in result.errors)
