# Design — Structured Log Analyzer

## Purpose

A CLI that samples a random batch of lines from a structured log CSV, lets an
LLM **explore the rest of the file with tools** to find the root cause, returns a
**schema-validated `LogAnalysis`**, and then **checks the model's cited evidence
against the source CSV** before handing it back. The same task can be run against
any of three providers — **Claude**, **Gemini**, or **NVIDIA NIM (Kimi K2)** —
chosen at runtime, so they can be compared on identical input.

## Design goals

1. **Provider-swappable** — change one config value (or pass `--provider`) to
   route the same prompt to a different LLM.
2. **One output contract** — every provider returns the same `LogAnalysis`
   Pydantic model, regardless of how that provider achieves structured output.
3. **Fair comparison** — the system prompt, user prompt, and sampled log batch
   are identical across providers; only the model call differs.
4. **Tool-augmented, but bounded** — tool-capable providers query the *whole*
   file via tools instead of stuffing it into the prompt, so log size doesn't
   inflate context; the loop is capped so a model can't run away.
5. **Grounded output** — the model's cited evidence is verified against the
   actual log rows, so a fluent-but-fabricated answer is flagged, not trusted.
6. **Pay only for what you use** — provider SDKs (and provider-specific tool code)
   are imported lazily, so you need just the SDK and key for the provider you run.

## High-level architecture

```
                    config.yaml + .env
                           │
                           ▼
                  ┌──────────────────┐
   CLI ─────────▶ │   main.py        │  parse args, load config/.env, --verbose
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │  analyzer.py     │  read → sample → build LogToolkit(rows)
                  │  run_analysis()  │  → analyze(text, toolkit) → validate
                  └───┬───────────┬──┘
            ┌─────────┘           └──────────┐
            ▼                                ▼
   ┌──────────────────┐            ┌────────────────────┐
   │  log_reader.py   │            │  providers/         │
   │  read_logs       │            │  get_provider(name) │  lazy registry
   │  sample_logs     │            └─────────┬───────────┘
   │  format_logs     │                      ▼
   └──────────────────┘   ┌──────────────────────────────────┐
                          │  LLMProvider (ABC) ─ base.py      │
        shared input      │  └ ToolCallingProvider (uses_tools)│
        ┌──────────────┐  └───────┬──────────┬──────────┬─────┘
        │  prompts.py  │◀───────┐ ▼          ▼          ▼
        │  models.py   │◀────┐  │ Claude     Nvidia     Gemini
        │  LogAnalysis │     │  │ (anthropic)(openai→NIM)(google-genai)
        └──────────────┘     │  │ │tool loop │tool loop │ plain
              ▲              │  │ └────┬─────┴────┬─────┘
   ┌──────────┴───────────┐  │  │      │ tool calls / results
   │  tools.py            │◀─┼──┼──────┘
   │  TOOL_SCHEMAS        │  │  │ tool_calling.py: to_anthropic_tools /
   │  LogToolkit.dispatch │  │  │ to_openai_tools · MAX_TOOL_ITERATIONS
   └──────────────────────┘  │  │
                             │  └──▶ all return a validated LogAnalysis
                             │              │
                             │              ▼
                    ┌────────┴─────────┐  ground cited LineIds against the CSV
                    │  validation.py   │  (whole file when tools_used) →
                    │ validate_analysis│  ValidationResult
                    └──────────────────┘
```

## Core flow

`run_analysis()` in [analyzer.py](log_analyzer/analyzer.py) is the spine. It is a
linear pipeline:

1. **Read** — [`read_logs`](log_analyzer/log_reader.py#L12) parses the structured
   CSV into a list of row dicts (`LineId`, `Time`, `Level`, `Content`, …).
2. **Sample** — [`sample_logs`](log_analyzer/log_reader.py#L22) picks a *random
   count* in `[min_size, max_size]`, then either a contiguous window (keeps
   correlated events together) or a random scatter, seedable for reproducibility.
3. **Format** — [`format_logs`](log_analyzer/log_reader.py#L40) renders the sampled
   rows as `[LineId] <Time> <Level> <Content>` so the model can cite line IDs.
4. **Toolkit** — a [`LogToolkit`](log_analyzer/tools.py) wraps the *full* CSV rows
   so tool-capable providers can query the whole file (not just the sample).
5. **Analyze** — [`get_provider`](log_analyzer/providers/__init__.py#L9)
   constructs the chosen provider and calls `.analyze(log_text, toolkit)`. For
   tool-capable providers this runs the two-phase tool loop (see below).
6. **Validate** — [`validate_analysis`](log_analyzer/validation.py) grounds the
   model's cited `evidence_line_ids` against the sampled rows and the full CSV,
   relaxed to the whole file when the provider used tools (see below).
7. **Return** — results are wrapped in an `AnalysisRun` dataclass (provider,
   model, sampled line IDs, count, the `LogAnalysis`, and the `ValidationResult`).

## Key abstractions

### `LLMProvider` (Strategy pattern) — [base.py](log_analyzer/providers/base.py)

An abstract base class defining the single seam every provider implements:

```python
def analyze(self, log_text: str, toolkit: LogToolkit) -> LogAnalysis: ...
```

It also centralizes the per-provider API-key lookup (`api_key_env` +
`_require_key()`), model resolution (`config.model_for(name)`), and a
`uses_tools` flag, so subclasses only contain provider-specific call logic.

**`ToolCallingProvider`** is a thin subclass that flips `uses_tools = True` — the
marker a provider opts into to run the tool loop. Plain providers (Gemini) extend
`LLMProvider` and ignore the `toolkit`; tool-capable ones (Claude, NVIDIA) extend
`ToolCallingProvider`. `analyzer` reads `provider.uses_tools` to pick the grounding
semantics. The toolkit is always passed, so the single `analyze(text, toolkit)`
seam serves both kinds.

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

| Provider | SDK | Structured output mechanism | Tools |
|---|---|---|---|
| **Claude** ([claude.py](log_analyzer/providers/claude.py)) | `anthropic` | Native `messages.parse(output_format=LogAnalysis)`; checks `stop_reason == "refusal"` | yes |
| **Gemini** ([gemini.py](log_analyzer/providers/gemini.py)) | `google-genai` | Native `response_schema=LogAnalysis` + `response_mime_type=application/json` | no (plain) |
| **NVIDIA** ([nvidia.py](log_analyzer/providers/nvidia.py)) | `openai` (OpenAI-compatible NIM endpoint) | **No native schema** — injects `schema_hint()` into the prompt + `response_format=json_object` | yes |

Claude and Gemini constrain the model natively; NVIDIA falls back to a
prompt-level contract because the OpenAI-compatible endpoint can't enforce an
arbitrary schema. All three converge on a validated `LogAnalysis`.

## Tool calling — [tools.py](log_analyzer/tools.py) + [tool_calling.py](log_analyzer/tool_calling.py)

Tool-capable providers don't just see the sampled lines — they investigate the
whole log through three tools before concluding:

| Tool | Purpose |
|---|---|
| `grep_logs(pattern, level?)` | Regex-search every line's Content, optional level filter |
| `get_window(center_line_id, n)` | The `n` lines either side of a LineId, for context |
| `count_by_level(start?, end?)` | Level histogram over an optional LineId range |

**Author once, adapt per provider.** The tools are defined a single time as plain
JSON Schema in `TOOL_SCHEMAS`, and executed by `LogToolkit.dispatch` against the
full CSV rows — both provider-neutral. Each provider only needs the *envelope*
translation, which lives in [tool_calling.py](log_analyzer/tool_calling.py):
`to_anthropic_tools` (→ `input_schema`) and `to_openai_tools` (→
`{"type":"function", ...}`). Adding Gemini later means a third `to_gemini_tools`,
not new tool logic.

**Two-phase loop** (same shape for both tool-capable providers):

1. **Explore** — a plain tool loop (Claude: `messages.create(tools=...)`; NVIDIA:
   `chat.completions.create(tools=...)`). The model requests tools, the provider
   runs them via `LogToolkit.dispatch` and feeds raw results back, repeating until
   the model stops asking — capped at `MAX_TOOL_ITERATIONS` rounds as a safety net.
2. **Conclude** — a final, tool-free call constrained to the schema (Claude:
   `messages.parse`; NVIDIA: `response_format=json_object` + schema in the prompt)
   that turns the exploration into the `LogAnalysis`.

The split keeps each step on a well-supported SDK path and dodges a real
constraint — Gemini (when added) can't combine function-calling with a response
schema in one call, so two phases is the portable template. Only small, capped
tool results enter context (`grep_logs` returns ≤ 50 rows with a `truncated`
flag), so **file size never inflates the prompt** — the scaling limit is memory
(the whole-file read) and tool recall, not the context window.

### Capability-aware thinking (Claude)

Adaptive thinking exists on the 4.6+ Opus/Sonnet family but is rejected by Haiku
4.5 and older models. Rather than hardcode a model list, `ClaudeProvider` asks the
Models API once at startup whether the configured model supports it and includes
`thinking: adaptive` only when it does (`**self._thinking`). This is what lets
`models.claude` be swapped between e.g. Haiku and Sonnet without code changes —
the default is Haiku 4.5 (cheapest tier) for cost-sensitive runs.

## Grounding validation — [validation.py](log_analyzer/validation.py)

Schema validation guarantees the response has the right *shape*; it says nothing
about whether the content is *true*. The model cites `evidence_line_ids` to back
its conclusion, so the cheapest meaningful accuracy check is to confirm those
citations are real — the model should only cite lines it was actually shown.

`validate_analysis(analysis, sampled_rows, all_rows, tools_used)` grades each
cited LineId into one of three buckets by comparing it to the source rows:

| Bucket | Condition | Interpretation |
|---|---|---|
| `grounded_line_ids` | in the sampled batch | Cited evidence the model was shown |
| `out_of_sample_line_ids` | in the full CSV, not in the sample | Referenced beyond the sample |
| `unknown_line_ids` | absent from the CSV entirely | Fabricated LineId |

Passing **both** the sampled rows and the full CSV is what lets it tell an
out-of-sample reference apart from a hallucination. Pass/fail then depends on
`tools_used` (set from `provider.uses_tools`):

- **No tools** — the model could only have seen the sample, so `out_of_sample`
  *and* `unknown` both fail. `is_valid` requires every citation grounded.
- **Tools** — the model can legitimately reach the whole file, so `out_of_sample`
  is accepted; only `unknown` (fabricated) fails.

Either way `is_valid` requires ≥1 citation; `issues` explains any failure.

This is deliberately a **non-blocking** check: a failed validation is attached to
the result and surfaced, but does not raise or change the exit code — the tool
reports the verdict rather than suppressing a suspect analysis. It catches
*fabricated* citations, not *irrelevant* ones (a real line that doesn't actually
support the claim) — a relevance check (e.g. an LLM-as-judge pass) is a natural
next layer.

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

[main.py](log_analyzer/main.py) prints run metadata to **stderr** (`provider`,
`model`, `tools=on/off`, sampled count, LineIds, and a `validation=PASSED/FAILED`
line with any issues) and a JSON object to **stdout** — so the structured result
can be piped while the run context stays visible. The stdout payload has two keys:

```json
{ "analysis": { ...LogAnalysis... }, "validation": { ...ValidationResult... } }
```

`--verbose` (`-v`) raises the `log_analyzer` logger to DEBUG on stderr, emitting a
step-by-step trace of the tool loop (each model call, `stop_reason`, the tools the
model requested, the results fed back, and — for NVIDIA — per-call token usage)
while keeping third-party loggers quiet. Errors are surfaced as a clean `Error: …`
message with a non-zero exit.

## Extension points

- **Add a plain provider:** subclass `LLMProvider`, implement `name` +
  `analyze(log_text, toolkit)` (ignore `toolkit`), set `api_key_env`, and add one
  branch to `get_provider()`. Reuse `SYSTEM_PROMPT` / `build_user_prompt()`.
- **Add a tool-capable provider:** subclass `ToolCallingProvider` instead, add a
  `to_<provider>_tools()` renderer in `tool_calling.py`, and run the two-phase loop
  — `claude.py` / `nvidia.py` are the templates. No change to `analyzer.py` or
  `validation.py`.
- **Change the analysis shape:** edit `LogAnalysis` once — Claude and Gemini pick
  up the new schema automatically; NVIDIA picks it up via `schema_hint()`.
- **Add/adjust a tool:** edit `TOOL_SCHEMAS` + the `LogToolkit` method once; every
  tool-capable provider gets it via the shared renderers.

## Trade-offs & limitations

- **In-process adapters, not a gateway.** Deliberate: the providers exploit
  provider-specific features (Claude's native parse, Gemini's response schema,
  each SDK's own tool-call format) that a uniform proxy would flatten. A gateway
  (LiteLLM, OpenRouter, Pydantic AI, …) becomes worth it only when cross-provider
  fallback, routing, or centralized cost/observability are needed.
- **Whole-file read.** `read_logs` loads the entire CSV into memory — fine for
  sample datasets, not multi-GB logs. This (not the context window) is the real
  ceiling as files grow, since tools keep prompt size bounded.
- **Tool recall is capped.** `grep_logs` returns ≤ 50 matches (with `truncated`),
  so on a huge file the model reasons over a slice; there's no pagination yet.
- **`MAX_TOOL_ITERATIONS` is a blunt cap.** The model isn't told the budget; it's
  a harness-side safety net, not a model-aware token/task budget.
- **Validation checks existence, not relevance.** A real-but-unsupportive citation
  passes; catching that needs a semantic/judge step.
- **Hand-maintained registry.** `get_provider` is explicit `if/elif`; no plugin
  auto-discovery (acceptable at this scale).
- **No retry/backoff layer** beyond what each SDK does by default.
