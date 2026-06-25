"""The provider-neutral agent loop.

One `run_agent_loop` drives the explore→conclude cycle for *every* tool-capable
provider. It never knows which provider it's driving — all Anthropic-vs-OpenAI
dialect lives behind the `AgentHooks` protocol (each method is a pure translation
the provider implements). The loop's own job is policy: enforce the budgets, run
the tool round-trips, and leave through exactly one named `StopReason`.

It returns a `LoopOutcome` carrying not just the answer but the *cost* — which
stop reason, how many steps, how many tokens — so downstream code (and a human
reading `--verbose`) can see *why* a run ended and what it spent.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .config import Budgets
from .models import LogAnalysis
from .tools import LogToolkit

log = logging.getLogger(__name__)


class StopReason(str, Enum):
    """The only doors out of the loop. Every exit is one of these — no silent break."""

    COMPLETED = "completed"  # got a valid structured LogAnalysis
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"  # hit max steps still wanting tools
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"  # crossed the token ceiling
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"  # crossed the wall-clock ceiling
    TRUNCATED = "truncated"  # a single response stopped on max_tokens
    REFUSED = "refused"  # model declined
    NO_PROGRESS = "no_progress"  # neither a tool call nor a final answer


@dataclass
class ToolCall:
    """A normalized tool request, dialect stripped off by `extract_tool_calls`."""

    id: str
    name: str
    args: dict


@dataclass
class LoopOutcome:
    """What the loop returns: the answer *and* the cost of getting (or not) there."""

    analysis: LogAnalysis | None
    stop_reason: StopReason
    steps_used: int
    tokens_used: int


class AgentHooks(Protocol):
    """The dialect seam. Each method translates one provider's wire format.

    Implemented by `ToolCallingProvider` subclasses; the loop only ever calls
    these — it contains no `if provider == ...`.
    """

    def seed_messages(self, log_text: str) -> list: ...
    def call_model(self, messages: list, use_tools: bool): ...
    def usage(self, response) -> tuple[int, int]: ...  # (input_tokens, output_tokens)
    def is_refusal(self, response) -> bool: ...
    def is_truncated(self, response) -> bool: ...
    def assistant_message(self, response): ...  # the turn to append to `messages`
    def extract_tool_calls(self, response) -> list[ToolCall]: ...
    def format_tool_results(self, results: list[tuple[ToolCall, str]]) -> list: ...
    def extract_final(self, response) -> LogAnalysis | None: ...
    def conclude_nudge(self): ...  # a user turn telling the model to wrap up


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_log_analysis(text: str | None) -> LogAnalysis | None:
    """Leniently parse a model's final text into a `LogAnalysis`, or `None`.

    Tolerates a markdown code fence and surrounding prose; returns `None` if no
    schema-valid JSON object can be found (the loop reads that as NO_PROGRESS).
    """
    if not text:
        return None
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = re.sub(r"^json\s*", "", body, flags=re.IGNORECASE).strip()
    try:
        return LogAnalysis.model_validate_json(body)
    except Exception:  # noqa: BLE001 - fall through to a looser search
        match = _JSON_RE.search(body)
        if match:
            try:
                return LogAnalysis.model_validate_json(match.group(0))
            except Exception:  # noqa: BLE001
                return None
    return None


def run_agent_loop(
    hooks: AgentHooks, toolkit: LogToolkit, log_text: str, budgets: Budgets
) -> LoopOutcome:
    """Drive the explore→conclude cycle under the budget policy.

    Best-effort at-limit policy: when a *budget* is exhausted or a response is
    truncated, make one final tool-free conclude call to salvage an answer from
    what's been gathered (its cost is added to `tokens_used`, so the overage is
    visible). A refusal or genuine no-progress returns no analysis — there's
    nothing to salvage. See design.md for the reasoning.
    """
    messages = hooks.seed_messages(log_text)
    steps = 0
    tokens = 0
    start = time.monotonic()
    log.debug(
        "agent loop start: budgets steps<=%d tokens<=%d seconds<=%s",
        budgets.max_steps,
        budgets.max_tokens,
        budgets.max_seconds,
    )

    def best_effort(reason: StopReason) -> LoopOutcome:
        """One final tool-free call to conclude with whatever we have."""
        nonlocal steps, tokens
        log.debug("%s -> best-effort conclude call", reason.value)
        messages.append(hooks.conclude_nudge())
        try:
            response = hooks.call_model(messages, use_tools=False)
        except Exception as exc:  # noqa: BLE001 - salvage failed; report the budget reason
            log.debug("best-effort conclude failed: %s", exc)
            return LoopOutcome(None, reason, steps, tokens)
        steps += 1
        i, o = hooks.usage(response)
        tokens += i + o
        analysis = hooks.extract_final(response)
        log.debug(
            "best-effort conclude: parsed=%s steps=%d tokens=%d",
            analysis is not None,
            steps,
            tokens,
        )
        return LoopOutcome(analysis, reason, steps, tokens)

    while True:
        # --- budget policy: checked every iteration, before the next call ---
        if steps >= budgets.max_steps:
            log.debug("step budget reached (%d) -> stopping", budgets.max_steps)
            return best_effort(StopReason.STEP_BUDGET_EXHAUSTED)
        if tokens >= budgets.max_tokens:
            log.debug("token budget reached (%d) -> stopping", budgets.max_tokens)
            return best_effort(StopReason.TOKEN_BUDGET_EXHAUSTED)
        if budgets.max_seconds is not None and time.monotonic() - start >= budgets.max_seconds:
            log.debug("time budget reached (%.1fs) -> stopping", budgets.max_seconds)
            return best_effort(StopReason.TIME_BUDGET_EXHAUSTED)

        # --- one model round-trip ---
        log.debug(
            "[step %d] calling model (steps_used=%d tokens_used=%d)",
            steps + 1,
            steps,
            tokens,
        )
        response = hooks.call_model(messages, use_tools=True)
        steps += 1
        i, o = hooks.usage(response)
        tokens += i + o
        log.debug("[step %d] usage in=%d out=%d running_tokens=%d", steps, i, o, tokens)

        # --- response-level terminal conditions ---
        if hooks.is_refusal(response):
            log.debug("[step %d] refusal", steps)
            return LoopOutcome(None, StopReason.REFUSED, steps, tokens)
        if hooks.is_truncated(response):
            log.debug("[step %d] response truncated on max_tokens", steps)
            return best_effort(StopReason.TRUNCATED)

        messages.append(hooks.assistant_message(response))

        # --- tool calls: execute and continue exploring ---
        calls = hooks.extract_tool_calls(response)
        if calls:
            log.debug("[step %d] tool calls: %s", steps, [(c.name, c.args) for c in calls])
            results = [(c, toolkit.dispatch(c.name, c.args)) for c in calls]
            for c, output in results:
                log.debug("[step %d] result for %s -> %s", steps, c.id, output)
            messages.extend(hooks.format_tool_results(results))
            continue

        # --- no tool calls: the model either answered, or it stalled ---
        final = hooks.extract_final(response)
        if final is not None:
            log.debug("[step %d] final analysis -> COMPLETED", steps)
            return LoopOutcome(final, StopReason.COMPLETED, steps, tokens)
        log.debug("[step %d] no tool call and no parseable answer -> NO_PROGRESS", steps)
        return LoopOutcome(None, StopReason.NO_PROGRESS, steps, tokens)
