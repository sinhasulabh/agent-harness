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
        └──────────────┘     │  │ └──hooks──┴──hooks──┘   plain
              ▲              │  │       │ (dialect)        │
              │              │  │       ▼                  │
   ┌──────────┴───────────┐  │  │ ┌──────────────────┐    │
   │  tools.py            │◀─┼──┼─│  agent_loop.py   │    │
   │  TOOL_SCHEMAS        │  │  │ │ run_agent_loop:  │    │
   │  LogToolkit.dispatch │  │  │ │ budgets · steps  │    │
   └──────────────────────┘  │  │ │ StopReason       │    │
   tool_calling.py: to_*_tools│  │ └────────┬─────────┘    │
                             │  │          │ LoopOutcome   │
                             │  └──────────┴───────┬───────┘
                             │   (analysis, stop_reason, steps, tokens)
                             │                     ▼
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
   constructs the chosen provider and calls `.analyze(log_text, toolkit)`, which
   returns a `LoopOutcome`. For tool-capable providers this runs the shared agent
   loop (see The agent loop, below); Gemini does a single native call.
6. **Validate** — when the outcome has an analysis,
   [`validate_analysis`](log_analyzer/validation.py) grounds the model's cited
   `evidence_line_ids` against the sampled rows and the full CSV, relaxed to the
   whole file when the provider used tools (see below).
7. **Return** — results are wrapped in an `AnalysisRun` dataclass: provider,
   model, sampled line IDs, count, the `LogAnalysis` (or `None`), the
   `ValidationResult` (or `None`), and the loop's cost — `stop_reason`,
   `steps_used`, `tokens_used`.

## Key abstractions

### `LLMProvider` (Strategy pattern) — [base.py](log_analyzer/providers/base.py)

An abstract base class defining the single seam every provider implements:

```python
def analyze(self, log_text: str, toolkit: LogToolkit) -> LoopOutcome: ...
```

It also centralizes the per-provider API-key lookup (`api_key_env` +
`_require_key()`), model resolution (`config.model_for(name)`), and a
`uses_tools` flag, so subclasses only contain provider-specific call logic.

**`ToolCallingProvider`** flips `uses_tools = True` and provides a **concrete**
`analyze` that just calls `run_agent_loop(self, ...)` — so a tool-capable provider
implements only the dialect hooks, never the loop. Plain providers (Gemini) extend
`LLMProvider` directly, ignore the `toolkit`, and return a single-step
`LoopOutcome`. `analyzer` reads `provider.uses_tools` to pick the grounding
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

The providers do **not** share a call shape — each speaks its own SDK dialect.
For the tool-capable two, that dialect is hidden behind the agent-loop hooks (see
below); for plain Gemini it's a single native-schema call.

| Provider | SDK | How the final `LogAnalysis` is produced | Tools |
|---|---|---|---|
| **Claude** ([claude.py](log_analyzer/providers/claude.py)) | `anthropic` | Agent loop; final answer parsed from the model's text (prompted to emit bare JSON) | yes |
| **Gemini** ([gemini.py](log_analyzer/providers/gemini.py)) | `google-genai` | Single call, native `response_schema=LogAnalysis` | no (plain) |
| **NVIDIA** ([nvidia.py](log_analyzer/providers/nvidia.py)) | `openai` (NIM endpoint) | Agent loop; final answer parsed from the model's text (schema in the prompt) | yes |

Gemini constrains the model natively in one shot. The tool-capable providers run
the shared agent loop and read the final answer out of the model's own text via
`parse_log_analysis` — a deliberate trade (give up per-provider native schema
enforcement) that lets the loop stay single-phase and identical across providers.
All three converge on a validated `LogAnalysis`.

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

Only small, capped tool results enter context (`grep_logs` returns ≤ 50 rows with
a `truncated` flag), so **file size never inflates the prompt** — the scaling
limit is memory (the whole-file read) and tool recall, not the context window.

## The agent loop — [agent_loop.py](log_analyzer/agent_loop.py)

One `run_agent_loop` drives the explore→conclude cycle for **every** tool-capable
provider. It contains the whole `while`-loop and **never knows which provider it
is driving** — there is no `if provider == ...` anywhere in it. All Anthropic-vs-
OpenAI difference collapses into a small set of hooks (`AgentHooks`) the provider
implements:

| Hook | What it translates |
|---|---|
| `seed_messages(log_text)` | the initial message list |
| `call_model(messages, use_tools)` | one model round-trip |
| `usage(response)` | `(input_tokens, output_tokens)` |
| `is_refusal` / `is_truncated(response)` | response-level terminal conditions |
| `assistant_message(response)` | the turn to append back |
| `extract_tool_calls(response)` | normalized `ToolCall`s |
| `format_tool_results(results)` | tool results → message(s) to append (batched; Anthropic wants one user message, OpenAI wants one `tool` message each) |
| `extract_final(response)` | a `LogAnalysis` parsed from the response, or `None` |
| `conclude_nudge()` | a user turn forcing a final answer |

### Budgets are a policy, not one integer — [config.py](log_analyzer/config.py)

Stop conditions are a small policy checked **every iteration, before the next
call**, configured under `budgets:` in `config.yaml`:

- **Step budget** (`max_steps`) — max model round-trips.
- **Token budget** (`max_tokens`) — cumulative input+output tokens across the
  whole loop. Each response's `usage` is accumulated; the loop stops when the
  running total crosses the ceiling. (This is the real-world lesson: a single
  integer step cap doesn't bound *spend*.)
- **Time budget** (`max_seconds`, optional) — wall-clock cap.

### Stop-reason taxonomy — no silent fall-through

Every exit is a named `StopReason` ([agent_loop.py](log_analyzer/agent_loop.py));
the loop can only leave through one of these doors, and each is explicit in code:

| `StopReason` | Meaning |
|---|---|
| `COMPLETED` | got a valid structured `LogAnalysis` |
| `STEP_BUDGET_EXHAUSTED` | hit `max_steps` still wanting tools |
| `TOKEN_BUDGET_EXHAUSTED` | crossed the token ceiling |
| `TIME_BUDGET_EXHAUSTED` | crossed the wall-clock ceiling |
| `TRUNCATED` | a single response stopped on `max_tokens` (named, not mistaken for "done") |
| `REFUSED` | model declined |
| `NO_PROGRESS` | neither a tool call nor a parseable final answer (a real stall) |

### Behavior at the limit — **best-effort** (decided on purpose)

When a *budget* is exhausted or a response is *truncated*, the loop makes **one
final tool-free conclude call** (`conclude_nudge` + `call_model(use_tools=False)`)
to salvage an answer from the evidence already gathered. `REFUSED` and
`NO_PROGRESS` get no salvage — a refusal won't be talked out of declining, and a
stalled model won't be unstuck by asking again — so they return `analysis=None`.

**Why best-effort over fail-loud:** the budgets exist to bound *cost*, not to
*want no answer*. By the time a budget trips, the model has usually gathered
useful evidence; one bounded conclude call turns that into a usable (if caveated)
analysis, while the `stop_reason` makes clear it was cut short. The cost of that
extra call is added to `tokens_used` and reported, so the overage past the token
budget is **visible, not hidden** — the honest version of "we went a little over
to give you an answer." (A `--strict` fail-loud mode would be a one-line policy
swap in the loop.)

### The loop reports cost, not just the answer

`run_agent_loop` returns a `LoopOutcome` — `analysis` (or `None`), `stop_reason`,
`steps_used`, `tokens_used`. `AnalysisRun` carries these through to the output
JSON's `run` block and the stderr summary, so a human (or a downstream scorer)
can see at a glance: *"stopped after 6 steps / 14k tokens because
STEP_BUDGET_EXHAUSTED."* This is the seam everything downstream reads.

### Capability-aware thinking (Claude)

Adaptive thinking exists on the 4.6+ Opus/Sonnet family but is rejected by Haiku
4.5 and older models. Rather than hardcode a model list, `ClaudeProvider` asks the
Models API once at startup whether the configured model supports it and includes
`thinking: adaptive` only when it does (`**self._thinking`). This is what lets
`models.claude` be swapped between e.g. Haiku (cheapest tier), Sonnet, and Opus
without code changes.

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
  (`min_size`/`max_size`/`strategy`/`seed`), per-provider `models`, `max_tokens`,
  and `budgets` (`max_steps`/`max_tokens`/`max_seconds` — the loop's stop policy).
- `load_config()` parses and **validates** (e.g. `max_size >= min_size`, strategy
  is `contiguous`/`random`, `budgets.max_steps >= 1`) and resolves `log_file`
  relative to the config file.
- **Secrets stay out of config** — API keys come from the environment / `.env`
  (loaded via `python-dotenv` in [main.py](log_analyzer/main.py#L36)), keyed per
  provider (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`).
- CLI flags (`--provider`, `--min-size`, `--max-size`) override config per run.

## Output

[main.py](log_analyzer/main.py) prints run metadata to **stderr** (`provider`,
`model`, `tools=on/off`, sampled count/LineIds, a `stop_reason=… steps=… tokens=…`
line, and `validation=PASSED/FAILED` with any issues) and a JSON object to
**stdout** — so the structured result can be piped while the run context stays
visible. The stdout payload has three keys (`analysis` and `validation` are
`null` when the loop produced no analysis, e.g. `REFUSED`):

```json
{
  "analysis":  { ...LogAnalysis... } | null,
  "validation":{ ...ValidationResult... } | null,
  "run": { "provider", "model", "tools_used", "stop_reason", "steps_used", "tokens_used" }
}
```

The `run` block is the cost/outcome seam — a human (or a downstream scorer) reads
`stop_reason` + `steps_used` + `tokens_used` to see why a run ended and what it
spent. `--verbose` (`-v`) raises the `log_analyzer` logger to DEBUG on stderr,
emitting the loop trace (per-step model calls, running token total, the tools the
model requested, results fed back, and the exit door taken) while keeping
third-party loggers quiet. Errors are a clean `Error: …` with a non-zero exit.

## Extension points

- **Add a plain provider:** subclass `LLMProvider`, implement `name` +
  `analyze(log_text, toolkit)` (ignore `toolkit`), set `api_key_env`, and add one
  branch to `get_provider()`. Reuse `SYSTEM_PROMPT` / `build_user_prompt()`.
- **Add a tool-capable provider:** subclass `ToolCallingProvider` and implement
  the dialect **hooks** (`call_model`, `usage`, `extract_tool_calls`,
  `extract_final`, …) + a `to_<provider>_tools()` renderer in `tool_calling.py`.
  You write **no loop** — `run_agent_loop` is reused as-is. `claude.py` /
  `nvidia.py` are the templates. No change to `agent_loop.py`, `analyzer.py`, or
  `validation.py`.
- **Change the analysis shape:** edit `LogAnalysis` once — Gemini picks it up via
  `response_schema`; the tool providers via the schema in their system prompt.
- **Add/adjust a tool:** edit `TOOL_SCHEMAS` + the `LogToolkit` method once; every
  tool-capable provider gets it via the shared renderers.
- **Change the stop policy:** tune `budgets` in `config.yaml`; swap best-effort for
  fail-loud in one place (`run_agent_loop`'s `best_effort`).

## Trade-offs & limitations

- **In-process adapters, not a gateway.** Deliberate: the providers exploit
  provider-specific features (Gemini's response schema, each SDK's own tool-call
  format) that a uniform proxy would flatten. A gateway (LiteLLM, OpenRouter,
  Pydantic AI, …) becomes worth it only when cross-provider fallback, routing, or
  centralized cost/observability are needed.
- **Prompt-driven final answer for the tool providers.** To keep one single-phase
  loop, Claude and NVIDIA are *asked* (via the system prompt) to emit bare JSON
  and the loop parses it leniently — trading native schema enforcement for a loop
  that's identical across providers. A malformed final answer becomes a named
  `NO_PROGRESS`, and the best-effort conclude gives a second chance.
- **Whole-file read.** `read_logs` loads the entire CSV into memory — fine for
  sample datasets, not multi-GB logs. This (not the context window) is the real
  ceiling as files grow, since tools keep prompt size bounded.
- **Tool recall is capped.** `grep_logs` returns ≤ 50 matches (with `truncated`),
  so on a huge file the model reasons over a slice; there's no pagination yet.
- **Budgets are harness-side, not model-aware.** The model isn't told its
  remaining step/token budget; the loop enforces it externally (a model-aware
  task budget would be a further step).
- **Validation checks existence, not relevance.** A real-but-unsupportive citation
  passes; catching that needs a semantic/judge step.
- **Hand-maintained registry.** `get_provider` is explicit `if/elif`; no plugin
  auto-discovery (acceptable at this scale).
- **No retry/backoff layer** beyond what each SDK does by default.
