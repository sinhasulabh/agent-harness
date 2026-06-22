# Log Analyzer

Reads a structured log CSV, sends a **random number of log lines** to an LLM, and
gets back a structured root-cause analysis as JSON:

```json
{
  "severity": "high",
  "suspected_cause": "mod_jk worker repeatedly entering error state",
  "evidence_line_ids": [1402, 1403, 1404],
  "next_step": "Restart the jk2 worker and check workers2.properties config"
}
```

You can switch between three providers:

| Provider | Model (default)             | SDK / endpoint                                  |
| -------- | --------------------------- | ----------------------------------------------- |
| `claude` | `claude-opus-4-8`           | Official Anthropic SDK (structured outputs)     |
| `gemini` | `gemini-2.5-flash`          | `google-genai` (native response schema)         |
| `nvidia` | `moonshotai/kimi-k2-instruct` | NVIDIA NIM, OpenAI-compatible endpoint (Kimi K2) |

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
provider: claude
```

The JSON analysis prints to stdout; a one-line run summary (provider, model, which
LineIds were sampled) prints to stderr, so you can pipe the JSON cleanly:

```bash
uv run log-analyzer --provider claude > analysis.json
```

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
├── analyzer.py          # read -> sample -> analyze orchestration
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

- The default `nvidia` model id is `moonshotai/kimi-k2-instruct` (Kimi K2 on NVIDIA
  NIM). If you have access to a different Kimi build/version, set its exact id under
  `models.nvidia` in `config.yaml`.
- `evidence_line_ids` refer to the `LineId` column in the CSV.
