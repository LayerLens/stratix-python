from __future__ import annotations

from layerlens.attestation import (
    HashChain,
    hmac_sign,
    hmac_verify,
    verify_trial,
)


class TestHMACSigning:
    def test_sign_produces_base64(self):
        sig = hmac_sign(b"test-key", b"sha256:" + b"a" * 64)
        assert sig  # non-empty string
        assert isinstance(sig, str)

    def test_sign_deterministic(self):
        data = b"sha256:" + b"a" * 64
        assert hmac_sign(b"test-key", data) == hmac_sign(b"test-key", data)

    def test_different_data_different_signatures(self):
        s1 = hmac_sign(b"test-key", b"sha256:" + b"a" * 64)
        s2 = hmac_sign(b"test-key", b"sha256:" + b"b" * 64)
        assert s1 != s2

    def test_different_keys_different_signatures(self):
        data = b"sha256:" + b"a" * 64
        assert hmac_sign(b"key-1", data) != hmac_sign(b"key-2", data)

    def test_verify_valid(self):
        data = b"sha256:" + b"a" * 64
        sig = hmac_sign(b"test-key", data)
        assert hmac_verify(b"test-key", data, sig)

    def test_verify_invalid(self):
        assert not hmac_verify(b"test-key", b"sha256:" + b"a" * 64, "bogus")

    def test_verify_wrong_data(self):
        sig = hmac_sign(b"test-key", b"sha256:" + b"a" * 64)
        assert not hmac_verify(b"test-key", b"sha256:" + b"b" * 64, sig)

    def test_verify_wrong_key(self):
        sig = hmac_sign(b"key-1", b"data")
        assert not hmac_verify(b"key-2", b"data", sig)


class TestUnsignedChainHasNoSignatures:
    def test_unsigned_chain_has_no_signatures(self):
        chain = HashChain()
        e1 = chain.add_event({"name": "span-1"})
        trial = chain.finalize()

        assert e1.signature is None
        assert e1.signing_key_id is None
        assert trial.signature is None

    def test_to_dict_omits_signature_when_unsigned(self):
        chain = HashChain()
        chain.add_event({"name": "span-1"})
        d = chain.to_dict()

        event = d["events"][0]
        assert "signature" not in event
        assert "signing_key_id" not in event


class TestVerifyTrialWithSigning:
    """Verify that verify_trial still works with externally-signed envelopes.

    In the server-side signing model, the backend signs the chain after
    ingestion. These tests simulate that by manually signing envelopes
    and verifying them with verify_trial().
    """

    def _build_and_sign(self, secret: bytes, key_id: str = "org-123"):
        """Build an unsigned chain, then manually sign each envelope."""
        chain = HashChain()
        chain.add_event({"name": "a"})
        chain.add_event({"name": "b"})
        envelopes = chain.envelopes
        trial = chain.finalize()

        # Simulate server-side signing
        for env in envelopes:
            env.signature = hmac_sign(secret, env.hash.encode("utf-8"))
            env.signing_key_id = key_id
        trial.signature = hmac_sign(secret, trial.hash.encode("utf-8"))
        trial.signing_key_id = key_id

        return envelopes, trial

    def test_valid_signed_trial(self):
        secret = b"test-key"
        envelopes, trial = self._build_and_sign(secret)

        result = verify_trial(envelopes, trial, signing_secret=secret)
        assert result.valid
        assert result.chain_valid
        assert result.trial_hash_valid
        assert result.signatures_valid
        assert result.errors == []

    def test_tampered_signature_detected(self):
        secret = b"test-key"
        envelopes, trial = self._build_and_sign(secret)

        # Tamper with the event signature
        envelopes[0].signature = "dGFtcGVyZWQ="  # base64("tampered")

        result = verify_trial(envelopes, trial, signing_secret=secret)
        assert not result.valid
        assert not result.signatures_valid
        assert result.chain_valid
        assert result.trial_hash_valid

    def test_wrong_key_rejects(self):
        envelopes, trial = self._build_and_sign(b"key-1")

        result = verify_trial(envelopes, trial, signing_secret=b"key-2")
        assert not result.valid
        assert not result.signatures_valid

    def test_unsigned_chain_passes_without_secret(self):
        """verify_trial without signing_secret ignores missing signatures."""
        chain = HashChain()
        chain.add_event({"name": "a"})
        envelopes = chain.envelopes
        trial = chain.finalize()

        result = verify_trial(envelopes, trial)
        assert result.valid
        assert result.signatures_valid  # vacuously true

    def test_stripped_signatures_detected(self):
        """When signing_secret is provided, missing signatures should fail."""
        secret = b"test-key"
        envelopes, trial = self._build_and_sign(secret)

        # Strip signatures
        envelopes[0].signature = None
        trial.signature = None

        result = verify_trial(envelopes, trial, signing_secret=secret)
        assert not result.valid
        assert not result.signatures_valid
        assert any("Missing signature" in e for e in result.errors)

    def test_single_event_signed_chain(self):
        """Signed chain with exactly one event works correctly."""
        secret = b"test-key"
        chain = HashChain()
        chain.add_event({"name": "only"})
        envelopes = chain.envelopes
        trial = chain.finalize()

        # Manually sign
        for env in envelopes:
            env.signature = hmac_sign(secret, env.hash.encode("utf-8"))
            env.signing_key_id = "org-1"
        trial.signature = hmac_sign(secret, trial.hash.encode("utf-8"))
        trial.signing_key_id = "org-1"

        assert len(envelopes) == 1
        assert envelopes[0].signature is not None

        result = verify_trial(envelopes, trial, signing_secret=secret)
        assert result.valid
        assert result.signatures_valid
