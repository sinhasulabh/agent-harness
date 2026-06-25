# Log Analyzer

Reads a structured log CSV, sends a **random number of log lines** to an LLM,
lets the model **explore the rest of the file with tools** to find the root
cause, then **validates its cited evidence against the source CSV** before
returning the result as JSON:

```json
{
  "analysis": {
    "severity": "high",
    "suspected_cause": "mod_jk worker repeatedly entering error state",
    "evidence_line_ids": [2, 9, 10, 11],
    "next_step": "Restart the jk2 worker and check workers2.properties config"
  },
  "validation": {
    "is_valid": true,
    "tools_used": true,
    "cited_line_ids": [2, 9, 10, 11],
    "grounded_line_ids": [2],
    "out_of_sample_line_ids": [9, 10, 11],
    "unknown_line_ids": [],
    "issues": []
  },
  "run": {
    "provider": "claude",
    "model": "claude-opus-4-8",
    "tools_used": true,
    "stop_reason": "completed",
    "steps_used": 3,
    "tokens_used": 6622
  }
}
```

The `run` block reports the **cost** of the run — why the loop stopped
(`stop_reason`) and what it spent (`steps_used`, `tokens_used`). When the loop
produces no analysis (e.g. the model refuses), `analysis` and `validation` are
`null` and `stop_reason` explains why.

You can switch between three providers:

| Provider | Model (default)        | SDK / endpoint                                  | Tool calling |
| -------- | ---------------------- | ----------------------------------------------- | ------------ |
| `claude` | `claude-opus-4-8`      | Official Anthropic SDK                          | **yes**      |
| `gemini` | `gemini-2.5-flash`     | `google-genai` (native response schema)         | no (planned) |
| `nvidia` | `moonshotai/kimi-k2.6` | NVIDIA NIM, OpenAI-compatible endpoint (Kimi K2) | **yes**      |

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
model, `tools=on/off`, sampled LineIds, and a `validation=PASSED/FAILED` line —
prints to stderr, so you can pipe the JSON cleanly:

```bash
uv run log-analyzer --provider claude > analysis.json
```

Add `--verbose` (`-v`) to log every step of the analysis — including the tool
loop — to stderr. Useful for understanding (or debugging) what the model did:

```bash
uv run log-analyzer --provider claude --verbose 2>run.log >analysis.json
grep "tool calls:" run.log    # just what the model asked for each round
```

## Tool calling

For tool-capable providers (`claude`, `nvidia`), the model isn't limited to the
sampled lines — it can investigate the **whole** log file through three tools
before concluding:

| Tool | What it does |
| ---- | ------------ |
| `grep_logs(pattern, level?)` | Regex-search every line's Content, optionally filtered by level |
| `get_window(center_line_id, n)` | Return the `n` lines before/after a LineId for context |
| `count_by_level(start?, end?)` | Level histogram, optionally over a LineId range |

A single shared loop ([agent_loop.py](log_analyzer/agent_loop.py)) drives every
tool-capable provider — the model calls tools, the loop runs them and feeds
results back, until the model emits a final answer. The provider-specific wire
format lives only in small per-provider *hooks*, so the loop itself is identical
for Claude and NVIDIA. The full file is loaded once and queried by the tools, so
log size doesn't inflate the prompt. Because the model can cite lines it
discovered beyond its sample, validation treats the whole CSV as fair evidence.

**Budgets and stop reasons.** The loop runs under a budget policy from
`config.yaml` — `max_steps` (round-trips), `max_tokens` (cumulative across the
loop), and optional `max_seconds` — checked before every model call. It always
exits through one **named** outcome (`stop_reason`): `completed`,
`step_budget_exhausted`, `token_budget_exhausted`, `truncated`, `refused`, or
`no_progress`. When a budget is hit (or a response is truncated) the loop makes
one final tool-free call to conclude with what it has (a deliberate *best-effort*
policy — see design.md); a refusal or genuine stall returns no analysis.

```yaml
budgets:
  max_steps: 6        # max model round-trips
  max_tokens: 60000   # cumulative input+output tokens across the loop
  max_seconds: null   # optional wall-clock cap (null = off)
```

## Validating the analysis

The model is asked to cite the `LineId` values that support its conclusion
(`evidence_line_ids`). After each run, those citations are checked against the
actual log rows so a fluent-but-fabricated answer doesn't slip through. Each
cited LineId is graded into one of three buckets:

| Bucket | Meaning |
| ------ | ------- |
| `grounded_line_ids` | Cited **and** present in the batch shown to the model |
| `out_of_sample_line_ids` | Exists in the CSV, but wasn't in the lines sent |
| `unknown_line_ids` | Not found anywhere in the CSV — a fabricated LineId |

What counts as valid depends on whether the model had tools (`tools_used`):

- **Without tools** — the model could only have seen the sampled lines, so an
  out-of-sample citation is suspect: `is_valid` requires every citation grounded.
- **With tools** — the model can legitimately cite lines it found via `grep_logs`
  / `get_window`, so out-of-sample citations are accepted; only `unknown_line_ids`
  (absent from the whole file = fabricated) fail.

Either way, `is_valid` also requires at least one citation. On failure, `issues`
explains why and the stderr summary shows `validation=FAILED`. The run still exits
`0` — the analysis is returned with the verdict attached rather than suppressed.

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
├── prompts.py           # shared system/user prompts (+ tool-mode suffix)
├── tools.py             # canonical tool schemas + LogToolkit (executes them on the CSV)
├── tool_calling.py      # per-provider tool rendering (to_anthropic_tools / to_openai_tools)
├── agent_loop.py        # the shared tool loop: budgets, StopReason, LoopOutcome
├── validation.py        # grounds the model's cited LineIds against the CSV
├── analyzer.py          # read -> sample -> analyze -> validate orchestration
├── main.py              # CLI entry point
└── providers/
    ├── base.py          # LLMProvider + ToolCallingProvider (dialect hooks)
    ├── claude.py        # Anthropic — agent-loop hooks
    ├── gemini.py        # Google Gemini — plain (no tools yet)
    └── nvidia.py        # NVIDIA NIM (Kimi K2) — agent-loop hooks
```

## Adding another provider

Subclass `LLMProvider` in `log_analyzer/providers/`, implement `name` and
`analyze(log_text, toolkit)` (return a `LoopOutcome`), add it to the factory in
`providers/__init__.py`, and add its model id under `models:` in `config.yaml`.

To give it **tool calling**, subclass `ToolCallingProvider` instead (sets
`uses_tools = True`): implement the dialect **hooks** (`call_model`, `usage`,
`extract_tool_calls`, `extract_final`, …) and add a `to_<provider>_tools()`
renderer in [tool_calling.py](log_analyzer/tool_calling.py). You write **no
loop** — the shared `run_agent_loop` is reused as-is. `claude.py` and `nvidia.py`
are the templates.

## Notes

- The default `nvidia` model id is `moonshotai/kimi-k2.6` (Kimi K2 on NVIDIA
  NIM). If you have access to a different Kimi build/version, set its exact id under
  `models.nvidia` in `config.yaml`.
- The Claude provider sends `thinking: adaptive` only when the model supports it
  (queried from the Models API at startup), so swapping `models.claude` between
  e.g. `claude-haiku-4-5` (no adaptive thinking) and `claude-sonnet-4-6` just works.
- `evidence_line_ids` refer to the `LineId` column in the CSV.
