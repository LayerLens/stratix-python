# CLI — Workflow Examples

Fifteen copy-paste workflows covering the most common CLI tasks.

---

## 1. Quick start: first trace to evaluation

Set up, list traces, inspect one, and evaluate it with a judge.

```bash
# Configure
export LAYERLENS_STRATIX_API_KEY="sk-..."

# See what traces are available
stratix trace list

# Inspect a specific trace
stratix trace get <TRACE_ID>

# Create a judge and test it against the trace
stratix judge create --name "Accuracy" --goal "Rate factual accuracy of the response" --model-id <MODEL_ID>
stratix judge test --judge-id <JUDGE_ID> --trace-id <TRACE_ID>
```

---

## 2. Run an evaluation and wait for results

```bash
# Run and block until done
stratix evaluate run \
  --model openai/gpt-4o \
  --benchmark arc-agi-2 \
  --wait

# Or fire and check later
stratix evaluate run --model openai/gpt-4o --benchmark arc-agi-2
stratix evaluate list --status in-progress
stratix evaluate get <EVAL_ID>
```

---

## 3. Compare models on the same benchmark

```bash
stratix evaluate run --model openai/gpt-4o --benchmark arc-agi-2 --wait
stratix evaluate run --model anthropic/claude-3-opus --benchmark arc-agi-2 --wait

# List results sorted by accuracy
stratix evaluate list --sort-by accuracy --order desc
```

---

## 4. Judge workflow: create, test, tune, apply

```bash
# Create a judge
stratix judge create \
  --name "Helpfulness" \
  --goal "Rate how helpful and actionable the response is on a 1-5 scale" \
  --model-id <MODEL_ID>

# Test against a sample trace
stratix judge test --judge-id <JUDGE_ID> --trace-id <TRACE_ID>

# Review the result
stratix --format json judge get <JUDGE_ID>

# Iterate: create a refined version
stratix judge create \
  --name "Helpfulness v2" \
  --goal "Rate helpfulness on a 1-5 scale with justification" \
  --model-id <MODEL_ID>
```

---

## 5. Trace search and export

```bash
# Search for traces matching a keyword
stratix trace search "customer support" --page-size 5

# Export a trace to a file
stratix trace export <TRACE_ID> -o trace_backup.json

# Export as JSON for piping
stratix --format json trace get <TRACE_ID> | jq '.id'
```

---

## 6. Bulk evaluation from a JSONL file

```bash
# Create a jobs file
cat > jobs.jsonl <<'EOF'
{"model": "openai/gpt-4o", "benchmark": "arc-agi-2"}
{"model": "openai/gpt-4o-mini", "benchmark": "arc-agi-2"}
{"model": "anthropic/claude-3-opus", "benchmark": "arc-agi-2"}
EOF

# Dry-run to preview
stratix bulk eval --file jobs.jsonl --dry-run

# Execute and wait
stratix bulk eval --file jobs.jsonl --wait
```

---

## 7. Bulk trace evaluation with a judge

```bash
# Create a trace ID file
stratix --format json trace list | jq -r '.[].id' > trace_ids.txt

# Dry-run to preview
stratix bulk eval \
  --judge-id <JUDGE_ID> \
  --traces trace_ids.txt \
  --dry-run

# Run trace evaluations for all traces
stratix bulk eval \
  --judge-id <JUDGE_ID> \
  --traces trace_ids.txt
```

---

## 8. CI/CD pipeline integration

```bash
# In your GitHub Actions workflow:
stratix evaluate run \
  --model openai/gpt-4o \
  --benchmark arc-agi-2 \
  --wait

# Generate a summary for the GitHub job
stratix ci report -o "$GITHUB_STEP_SUMMARY"

# Or output JSON for custom processing
stratix ci report -o report.json
```

---

## 9. Integration monitoring

```bash
# List all integrations
stratix integration list

# Test a specific integration
stratix integration test <INTEGRATION_ID>

# JSON output for scripting
stratix --format json integration list | jq '.[] | select(.status != "active")'
```

---

## 10. Scorer management

```bash
# List existing scorers
stratix scorer list

# Create a scorer with dry-run
stratix scorer create \
  --name "Code Quality" \
  --description "Evaluates generated code for correctness, readability, and best practices" \
  --model-id <MODEL_ID> \
  --prompt "Score the following code on a 1-10 scale for quality..." \
  --dry-run

# Create for real
stratix scorer create \
  --name "Code Quality" \
  --description "Evaluates generated code for correctness, readability, and best practices" \
  --model-id <MODEL_ID> \
  --prompt "Score the following code on a 1-10 scale for quality..."

# Delete with confirmation
stratix scorer delete <SCORER_ID>

# Delete without prompt
stratix scorer delete <SCORER_ID> -y
```

---

## 11. Evaluation spaces

```bash
# List spaces
stratix space list

# Create a private space
stratix space create \
  --name "Q1 Model Comparison" \
  --description "Comparing GPT-4o vs Claude 3 Opus for Q1 release" \
  --visibility private

# Create a public space (dry-run first)
stratix space create \
  --name "Public Leaderboard" \
  --visibility public \
  --dry-run

# Get space details by slug or ID
stratix space get q1-model-comparison

# Clean up
stratix space delete <SPACE_ID> -y
```

---

## 12. JSON output and scripting

```bash
# Pipe trace IDs into a loop
stratix --format json trace list | jq -r '.[].id' | while read id; do
  echo "Exporting $id..."
  stratix trace export "$id" -o "traces/${id}.json"
done

# Get evaluation accuracy as a number
ACCURACY=$(stratix --format json evaluate get <EVAL_ID> | jq -r '.accuracy')
echo "Accuracy: $ACCURACY"

# Filter evaluations by status
stratix --format json evaluate list | jq '[.[] | select(.status == "success")]'
```

---

## 13. Pagination and sorting

```bash
# Page through traces
stratix trace list --page 1 --page-size 20
stratix trace list --page 2 --page-size 20

# Sort evaluations
stratix evaluate list --sort-by accuracy --order desc --page-size 5
stratix evaluate list --sort-by submitted_at --order asc

# Sort spaces
stratix space list --sort-by created_at --order desc
```

---

## 14. Verbose mode and debugging

```bash
# Enable verbose output to see HTTP requests
stratix -v trace list

# Combine with JSON output
stratix -v --format json evaluate get <EVAL_ID>

# Debug authentication issues
stratix -v integration list
```

---

## 15. Clean up resources

```bash
# Delete a trace (with confirmation prompt)
stratix trace delete <TRACE_ID>

# Delete without prompting
stratix trace delete <TRACE_ID> -y

# Delete a scorer (dry-run first)
stratix scorer delete <SCORER_ID> --dry-run
stratix scorer delete <SCORER_ID> -y

# Delete a space
stratix space delete <SPACE_ID> -y
```
