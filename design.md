# Design — Structured Log Analyzer

## Purpose

A CLI that samples a random batch of lines from a structured log CSV and asks
an LLM to return a **schema-validated root-cause analysis**. The same task can
be run against any of three providers — **Claude**, **Gemini**, or **NVIDIA NIM
(Kimi K2)** — chosen at runtime, so they can be compared on identical input.

## Design goals

1. **Provider-swappable** — change one config value (or pass `--provider`) to
   route the same prompt to a different LLM.
2. **One output contract** — every provider returns the same `LogAnalysis`
   Pydantic model, regardless of how that provider achieves structured output.
3. **Fair comparison** — the system prompt, user prompt, and sampled log batch
   are identical across providers; only the model call differs.
4. **Pay only for what you use** — provider SDKs are imported lazily, so you
   need just the SDK (and key) for the provider you actually run.

## High-level architecture

```
                    config.yaml + .env
                           │
                           ▼
                  ┌──────────────────┐
   CLI ─────────▶ │   main.py        │  parse args, load config, load .env
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │  analyzer.py     │  orchestrates: read → sample → analyze
                  │  run_analysis()  │  returns AnalysisRun
                  └───┬───────────┬──┘
            ┌─────────┘           └──────────┐
            ▼                                ▼
   ┌──────────────────┐            ┌────────────────────┐
   │  log_reader.py   │            │  providers/         │
   │  read_logs       │            │  get_provider(name) │  lazy registry
   │  sample_logs     │            └─────────┬───────────┘
   │  format_logs     │                      ▼
   └──────────────────┘            ┌────────────────────┐
                                   │  LLMProvider (ABC)  │  base.py
            shared input           │  .analyze(text) ─┐  │
            ┌──────────────┐       └──────────────────┼──┘
            │  prompts.py  │◀─────────┐    ┌───────────┴───────────┐
            │  models.py   │◀──────┐  │    ▼          ▼            ▼
            │  LogAnalysis │       │  │ ClaudeProvider GeminiProvider NvidiaProvider
            └──────────────┘       │  │ (anthropic)   (google-genai) (openai→NIM)
                  ▲                │  │    │          │            │
                  └────────────────┴──┴────┴──────────┴────────────┘
                       all return a validated LogAnalysis
```

## Core flow

`run_analysis()` in [analyzer.py](log_analyzer/analyzer.py) is the spine. It is a
linear pipeline:

1. **Read** — [`read_logs`](log_analyzer/log_reader.py#L12) parses the structured
   CSV into a list of row dicts (`LineId`, `Time`, `Level`, `Content`, …).
2. **Sample** — [`sample_logs`](log_analyzer/log_reader.py#L22) picks a *random
   count* in `[min_size, max_size]`, then either a contiguous window (keeps
   correlated events together) or a random scatter, seedable for reproducibility.
3. **Format** — [`format_logs`](log_analyzer/log_reader.py#L40) renders rows as
   `[LineId] <Time> <Level> <Content>` so the model can cite line IDs.
4. **Analyze** — [`get_provider`](log_analyzer/providers/__init__.py#L9)
   constructs the chosen provider and calls `.analyze(log_text)`.
5. **Return** — results are wrapped in an `AnalysisRun` dataclass (provider,
   model, sampled line IDs, count, and the `LogAnalysis`).

## Key abstractions

### `LLMProvider` (Strategy pattern) — [base.py](log_analyzer/providers/base.py)

An abstract base class defining the single seam every provider implements:

```python
def analyze(self, log_text: str) -> LogAnalysis: ...
```

It also centralizes the per-provider API-key lookup (`api_key_env` +
`_require_key()`) and model resolution (`config.model_for(name)`), so subclasses
only contain provider-specific call logic.

### Provider registry (lazy) — [providers/__init__.py](log_analyzer/providers/__init__.py)

`get_provider(name, config)` maps a name to a concrete provider and imports its
SDK **inside the branch**. Running with `provider: gemini` never imports
`anthropic` or `openai`. The trade-off: provider wiring is a hand-maintained
`if/elif` rather than auto-discovery — fine at three providers.

### The shared contract — [models.py](log_analyzer/models.py) + [prompts.py](log_analyzer/prompts.py)

`LogAnalysis` (severity, suspected_cause, evidence_line_ids, next_step) is the
**single source of truth** for the output shape. `SYSTEM_PROMPT` and
`build_user_prompt()` are shared verbatim. This is what makes cross-provider
comparison fair — input and output schema are constant; only the model differs.

## How each provider satisfies the same contract

This is the most important design point: the providers do **not** share a call
shape, because each has a different structured-output mechanism. The `LLMProvider`
seam hides that divergence.

| Provider | SDK | Structured output mechanism | Validation |
|---|---|---|---|
| **Claude** ([claude.py](log_analyzer/providers/claude.py)) | `anthropic` | Native `messages.parse(output_format=LogAnalysis)` + `thinking: adaptive`; checks `stop_reason == "refusal"` | SDK returns `parsed_output` |
| **Gemini** ([gemini.py](log_analyzer/providers/gemini.py)) | `google-genai` | Native `response_schema=LogAnalysis` + `response_mime_type=application/json` | SDK `response.parsed`, falls back to validating raw text |
| **NVIDIA** ([nvidia.py](log_analyzer/providers/nvidia.py)) | `openai` (OpenAI-compatible NIM endpoint) | **No native schema** — injects `schema_hint()` into the prompt + `response_format=json_object` | Manual `LogAnalysis.model_validate_json()` |

Claude and Gemini constrain the model natively; NVIDIA falls back to a
prompt-level contract because the OpenAI-compatible endpoint can't enforce an
arbitrary schema. All three converge on a validated `LogAnalysis`.

## Configuration — [config.py](log_analyzer/config.py) + `config.yaml`

- `config.yaml` holds non-secret settings: `provider`, `log_file`, `sampling`
  (`min_size`/`max_size`/`strategy`/`seed`), per-provider `models`, `max_tokens`.
- `load_config()` parses and **validates** (e.g. `max_size >= min_size`,
  strategy is `contiguous`/`random`) and resolves `log_file` relative to the
  config file.
- **Secrets stay out of config** — API keys come from the environment / `.env`
  (loaded via `python-dotenv` in [main.py](log_analyzer/main.py#L36)), keyed per
  provider (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`).
- CLI flags (`--provider`, `--min-size`, `--max-size`) override config per run.

## Output

[main.py](log_analyzer/main.py) prints a metadata line to **stderr**
(`provider`, `model`, sampled count, LineIds) and the JSON analysis to
**stdout** — so the structured result can be piped while the run context stays
visible. Errors are surfaced as a clean `Error: …` message with a non-zero exit.

## Extension points

- **Add a provider:** subclass `LLMProvider`, implement `name` + `analyze`, set
  `api_key_env`, and add one branch to `get_provider()`. Reuse `SYSTEM_PROMPT` /
  `build_user_prompt()` so it stays comparable.
- **Change the analysis shape:** edit `LogAnalysis` once — Claude and Gemini pick
  up the new schema automatically; NVIDIA picks it up via `schema_hint()`.
- **New sampling strategy:** extend `SamplingConfig.strategy` and `sample_logs()`.

## Trade-offs & limitations

- **In-process adapters, not a gateway.** Deliberate: the providers exploit
  provider-specific features (Claude's native parse + adaptive thinking, Gemini's
  response schema) that a uniform proxy would flatten. A gateway (LiteLLM,
  OpenRouter, Pydantic AI, …) becomes worth it only when cross-provider
  fallback, routing, or centralized cost/observability are needed.
- **Whole-file read.** `read_logs` loads the entire CSV into memory — fine for
  sample datasets, not for multi-GB logs.
- **Hand-maintained registry.** `get_provider` is explicit `if/elif`; no plugin
  auto-discovery (acceptable at this scale).
- **No retry/backoff layer** beyond what each SDK does by default.
