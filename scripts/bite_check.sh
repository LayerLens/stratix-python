#!/usr/bin/env bash
#
# Agent-safety bite-check (LAY-3624 / §5). A new or changed test that still
# PASSES when its target guard is mutated guards nothing — this is how an agent
# (or a human) ships a vacuous test. This gate runs mutation ONLY on the tier-1
# source modules a PR actually changed, scoped to the relevant tests, and FAILS
# if the per-module kill-rate is below its ratcheted floor (a vacuous/weakened
# test drops the score). It is the PR-time complement to the nightly full-module
# mutation job (scripts/mutation_tier1.sh) — fast because it touches only the
# changed modules.
#
# Usage:  BASE_REF=origin/main bash scripts/bite_check.sh
# Requires: mutmut==2.5.1, pytest, pytest-cov, an editable install, sqlite3.
#
set -uo pipefail

PY="${PYTHON:-python}"
MUT="${MUTMUT:-mutmut}"
PT="$PY -m pytest -x -q --no-header -p no:cacheprovider --assert=plain"
BASE_REF="${BASE_REF:-origin/main}"
SRC=src/layerlens/instrument
ASRC=src/layerlens/attestation
PKG=layerlens.instrument

# module-file | dotted module | scoped tests | per-module FLOOR (% killed, logic-dominated).
# Floors are the ratchet — raised as coverage improves; pricing is EXCLUDED (its
# bundled rate table dominates the mutant count, so its score is reference-data,
# not logic — the nightly job reports it but it is not a PR gate).
MODULES=(
  "$SRC/_secret_scrub.py|$PKG._secret_scrub|tests/instrument/test_secret_scrub.py tests/instrument/adapters/protocols/test_no_content_sweep.py|50"
  "$SRC/_capture_config.py|$PKG._capture_config|tests/instrument/test_capture_config.py tests/instrument/test_redaction_backstop.py tests/instrument/test_content_keys_guard.py tests/instrument/test_layer_suppression.py tests/instrument/test_event_schema.py|50"
  "$SRC/_collector.py|$PKG._collector|tests/instrument/test_redaction_backstop.py tests/instrument/test_secret_scrub.py tests/instrument/test_trace_context.py tests/instrument/test_cost_chokepoint.py tests/instrument/test_attestation_quarantine.py tests/instrument/test_collector_seam.py tests/attestation/test_integration.py|50"
  "$SRC/_spend_ledger.py|$PKG._spend_ledger|tests/instrument/adapters/protocols/test_ucp_invariants.py|50"
  "$SRC/_upload.py|$PKG._upload|tests/instrument/test_upload.py|50"
  "$ASRC/_chain.py|layerlens.attestation._chain|tests/attestation/|50"
  "$ASRC/_verify.py|layerlens.attestation._verify|tests/attestation/|50"
)

changed_files() {
  git diff --name-only "$BASE_REF"...HEAD 2>/dev/null || git diff --name-only HEAD~1 2>/dev/null || true
}

CHANGED="$(changed_files)"
echo "Bite-check vs $BASE_REF — changed files:"; echo "$CHANGED" | sed 's/^/  /'

rm -f .mutmut-cache
fail=0
ran=0
for entry in "${MODULES[@]}"; do
  IFS='|' read -r file dotted tests floor <<< "$entry"
  # Gate a module only when the PR touched the module OR its tests.
  touched=0
  echo "$CHANGED" | grep -qF "$file" && touched=1
  for t in $tests; do echo "$CHANGED" | grep -qF "${t%/}" && touched=1; done
  [ "$touched" -eq 0 ] && continue
  ran=$((ran+1))
  echo "::group::bite-check $file (floor ${floor}%)"
  rm -f .coverage
  if ! $PY -m pytest -q -p no:cacheprovider --no-header --cov="$dotted" --cov-report= $tests </dev/null > "/tmp/bite_$(basename "$file").log" 2>&1; then
    echo "::error title=bite-check $file::scoped tests are not green — fix before merging"; fail=1; echo "::endgroup::"; continue
  fi
  $MUT run --paths-to-mutate "$file" --runner "$PT $tests" --use-coverage --simple-output --no-progress --CI </dev/null > "/tmp/bite_$(basename "$file").mut.log" 2>&1 || true
  read -r k s < <(sqlite3 .mutmut-cache "SELECT \
    sum(case when status='ok_killed' then cnt else 0 end), \
    sum(case when status='bad_survived' then cnt else 0 end) \
    FROM (SELECT m.status status, count(*) cnt FROM Mutant m JOIN Line l ON m.line=l.id JOIN SourceFile sf ON l.sourcefile=sf.id WHERE sf.filename='$file' GROUP BY m.status);" 2>/dev/null)
  k=${k:-0}; s=${s:-0}; tot=$((k + s))
  if [ "$tot" -eq 0 ]; then echo "::warning::no mutants for $file"; echo "::endgroup::"; continue; fi
  score=$((100 * k / tot))
  echo "$file: killed=$k survived=$s score=${score}% (floor ${floor}%)"
  if [ "$score" -lt "$floor" ]; then
    echo "::error title=bite-check FLOOR $file::mutation score ${score}% < floor ${floor}% — a changed test does not bite its guard (vacuous/weakened test rejected)"
    fail=1
  fi
  rm -f .mutmut-cache
  echo "::endgroup::"
done

[ "$ran" -eq 0 ] && echo "No tier-1 module changed — bite-check skipped."
exit $fail
