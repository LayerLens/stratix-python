#!/usr/bin/env bash
#
# Tier-1 mutation score (LAY-3572). Runs `mutmut` per important-corner module
# with a SCOPED test runner (so each mutant runs only the relevant tests and the
# job stays tractable) and reports a per-module mutation score + surviving-mutant
# list. REPORT-ONLY by default: it emits ::warning:: for modules below the soft
# floor but exits 0, so the nightly job never red-flags on a flaky baseline.
# Set MUTATION_ENFORCE=1 to make it exit non-zero below the floor (future ratchet).
#
# Requires: mutmut (2.5.x — parso-based, no Rust build), pytest, pytest-cov,
# sqlite3, and an editable install of the package. The nightly workflow installs
# these. Run locally with:  bash scripts/mutation_tier1.sh
#
set -uo pipefail

PY="${PYTHON:-python}"
MUT="${MUTMUT:-mutmut}"
PT="$PY -m pytest -x -q --no-header -p no:cacheprovider --assert=plain"
FLOOR="${MUTATION_FLOOR:-50}"          # soft per-module floor (% killed)
ENFORCE="${MUTATION_ENFORCE:-0}"
SRC=src/layerlens/instrument
ASRC=src/layerlens/attestation
PKG=layerlens.instrument

# name | file(to mutate) | dotted module(for --cov) | scoped tests
MODULES=(
  "pricing|$SRC/adapters/providers/pricing.py|$PKG.adapters.providers.pricing|tests/instrument/adapters/providers/test_pricing.py tests/instrument/adapters/frameworks/test_cost_usd_fire.py"
  "secret_scrub|$SRC/_secret_scrub.py|$PKG._secret_scrub|tests/instrument/test_secret_scrub.py tests/instrument/adapters/protocols/test_no_content_sweep.py"
  "capture_config|$SRC/_capture_config.py|$PKG._capture_config|tests/instrument/test_capture_config.py tests/instrument/test_redaction_backstop.py tests/instrument/test_content_keys_guard.py tests/instrument/test_layer_suppression.py tests/instrument/test_event_schema.py tests/instrument/adapters/protocols/test_no_content_sweep.py tests/instrument/adapters/protocols/test_protocol_redaction.py"
  "collector|$SRC/_collector.py|$PKG._collector|tests/instrument/test_redaction_backstop.py tests/instrument/adapters/protocols/test_no_content_sweep.py tests/instrument/test_secret_scrub.py tests/instrument/test_trace_context.py tests/attestation/test_integration.py"
  "upload|$SRC/_upload.py|$PKG._upload|tests/instrument/test_upload.py"
  "chain|$ASRC/_chain.py|layerlens.attestation._chain|tests/attestation/"
  "verify|$ASRC/_verify.py|layerlens.attestation._verify|tests/attestation/"
  "ap2|$SRC/adapters/protocols/ap2.py|$PKG.adapters.protocols.ap2|tests/instrument/adapters/protocols/test_payment_guardrails.py tests/instrument/adapters/protocols/test_protocol_redaction.py tests/instrument/adapters/protocols/test_no_content_sweep.py tests/instrument/adapters/protocols/test_certification.py tests/instrument/adapters/protocols/test_base_protocol.py"
  "a2ui|$SRC/adapters/protocols/a2ui.py|$PKG.adapters.protocols.a2ui|tests/instrument/adapters/protocols/test_protocol_redaction.py tests/instrument/adapters/protocols/test_certification.py tests/instrument/adapters/protocols/test_no_content_sweep.py"
  "ucp|$SRC/adapters/protocols/ucp.py|$PKG.adapters.protocols.ucp|tests/instrument/adapters/protocols/test_protocol_redaction.py tests/instrument/adapters/protocols/test_no_content_sweep.py tests/instrument/adapters/protocols/test_certification.py"
  "emit_helpers|$SRC/adapters/providers/_emit_helpers.py|$PKG.adapters.providers._emit_helpers|tests/instrument/adapters/providers/"
)

rm -f .mutmut-cache

for entry in "${MODULES[@]}"; do
  IFS='|' read -r name file dotted tests <<< "$entry"
  echo "::group::mutation $name"
  rm -f .coverage
  if ! $PY -m pytest -q -p no:cacheprovider --no-header --cov="$dotted" --cov-report= $tests </dev/null > "/tmp/mut_$name.pre.log" 2>&1; then
    echo "::warning title=mutation $name::precheck failed (scoped tests not green) — skipping"
    echo "::endgroup::"; continue
  fi
  $MUT run --paths-to-mutate "$file" --runner "$PT $tests" --use-coverage --simple-output --no-progress --CI </dev/null > "/tmp/mut_$name.run.log" 2>&1 || true
  echo "::endgroup::"
done

# Per-module score from the accumulated cache (read fully before looping so the
# enforce-fail flag lives in THIS shell, not a pipe subshell).
declare -A killed survived
while IFS='|' read -r fname status cnt; do
  [ -z "${fname:-}" ] && continue
  case "$status" in
    ok_killed) killed["$fname"]=$cnt ;;
    bad_survived) survived["$fname"]=$cnt ;;
  esac
done < <(sqlite3 .mutmut-cache "SELECT sf.filename, m.status, count(*) FROM Mutant m JOIN Line l ON m.line=l.id JOIN SourceFile sf ON l.sourcefile=sf.id WHERE m.status IN ('ok_killed','bad_survived') GROUP BY sf.filename, m.status;")

fail=0
{
  echo "### Tier-1 mutation score (LAY-3572)"
  echo ""
  echo "| module | killed | survived | score |"
  echo "| --- | ---: | ---: | ---: |"
} | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

for f in $(printf '%s\n' "${!killed[@]}" "${!survived[@]}" | sort -u); do
  k=${killed[$f]:-0}; s=${survived[$f]:-0}; tot=$((k + s))
  [ "$tot" -eq 0 ] && continue
  score=$((100 * k / tot))
  printf '| %s | %d | %d | %d%% |\n' "$f" "$k" "$s" "$score" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
  if [ "$score" -lt "$FLOOR" ]; then
    echo "::warning title=mutation floor::$f mutation score ${score}% < floor ${FLOOR}%"
    [ "$ENFORCE" = "1" ] && fail=1
  fi
done

exit $fail
