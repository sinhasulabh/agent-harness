# Log Analyzer

Reads a structured log CSV, sends a **random number of log lines** to an LLM,
gets back a structured root-cause analysis, and **validates the model's cited
evidence against the source CSV** before returning it as JSON:

```json
{
  "analysis": {
    "severity": "high",
    "suspected_cause": "mod_jk worker repeatedly entering error state",
    "evidence_line_ids": [1402, 1403, 1404],
    "next_step": "Restart the jk2 worker and check workers2.properties config"
  },
  "validation": {
    "is_valid": true,
    "cited_line_ids": [1402, 1403, 1404],
    "grounded_line_ids": [1402, 1403, 1404],
    "out_of_sample_line_ids": [],
    "unknown_line_ids": [],
    "issues": []
  }
}
```

You can switch between three providers:

| Provider | Model (default)             | SDK / endpoint                                  |
| -------- | --------------------------- | ----------------------------------------------- |
| `claude` | `claude-opus-4-8`           | Official Anthropic SDK (structured outputs)     |
| `gemini` | `gemini-2.5-flash`          | `google-genai` (native response schema)         |
| `nvidia` | `moonshotai/kimi-k2.6` | NVIDIA NIM, OpenAI-compatible endpoint (Kimi K2) |

## Setup (uv)

```bash
uv sync                      # create the venv and install dependencies
cp .env.example .env         # add the API key(s) for the provider(s) you use
```

`.env` keys (only the selected provider's key is required):

```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
NVIDIA_API_KEY=nvapi-...
```

## Run

```bash
uv run log-analyzer
```

Switch the provider for a single run (no config edit needed):

```bash
uv run log-analyzer --provider gemini
uv run log-analyzer --provider nvidia
```

Or set the default permanently in `config.yaml`:

```yaml
provider: nvidia
```

The JSON (analysis + validation) prints to stdout; a short run summary — provider,
model, sampled LineIds, and a `validation=PASSED/FAILED` line — prints to stderr,
so you can pipe the JSON cleanly:

```bash
uv run log-analyzer --provider claude > analysis.json
```

## Validating the analysis

The model is asked to cite the `LineId` values that support its conclusion
(`evidence_line_ids`). After each run, those citations are checked against the
actual log rows so a fluent-but-fabricated answer doesn't slip through. Each
cited LineId is graded into one of three buckets:

| Bucket | Meaning |
| ------ | ------- |
| `grounded_line_ids` | Cited **and** present in the batch shown to the model — what we want |
| `out_of_sample_line_ids` | Exists in the CSV, but wasn't in the lines sent — the model reached beyond its input |
| `unknown_line_ids` | Not found anywhere in the CSV — a fabricated LineId |

`is_valid` is `true` only when the model cited at least one line and **every**
citation is grounded. On failure, `issues` explains why, and the stderr summary
shows `validation=FAILED`. The run still exits `0` — the analysis is returned
with the validation verdict attached rather than being suppressed.

## Configuring the sample size

The number of log lines sent each run is a **random count** drawn from the range in
`config.yaml`. Adjust the range to control how much data is analyzed:

```yaml
sampling:
  min_size: 1
  max_size: 25
  strategy: contiguous   # contiguous = random consecutive window; random = random scatter
  seed: null             # set an int for reproducible sampling
```

You can also override the range per run:

```bash
uv run log-analyzer --min-size 10 --max-size 50
```

## Project layout

```
config.yaml              # provider, sample size, model ids, log file path
pyproject.toml           # uv / dependency manifest
log_analyzer/
├── config.py            # loads config.yaml + resolves the log path
├── log_reader.py        # reads the CSV and samples a random batch
├── models.py            # LogAnalysis — the shared response schema (Pydantic)
├── prompts.py           # shared system/user prompts
├── validation.py        # grounds the model's cited LineIds against the CSV
├── analyzer.py          # read -> sample -> analyze -> validate orchestration
├── main.py              # CLI entry point
└── providers/
    ├── base.py          # LLMProvider interface
    ├── claude.py        # Anthropic
    ├── gemini.py        # Google Gemini
    └── nvidia.py        # NVIDIA NIM (Kimi K2)
```

## Adding another provider

Subclass `LLMProvider` in `log_analyzer/providers/`, implement `name` and
`analyze()` (return a `LogAnalysis`), add it to the factory in
`providers/__init__.py`, and add its model id under `models:` in `config.yaml`.

## Notes

- The default `nvidia` model id is `moonshotai/kimi-k2.6` (Kimi K2 on NVIDIA
  NIM). If you have access to a different Kimi build/version, set its exact id under
  `models.nvidia` in `config.yaml`.
- `evidence_line_ids` refer to the `LineId` column in the CSV.
