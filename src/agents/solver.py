"""The solver loop: search the web, write the solutions, check them, repeat until the answer passes.

    supervisor ──> solve ──(tool call)──> tools ──> solve ...
                     │
                     └─(final answer)──> verify ──(rejected, rounds left)──> solve
                                            └─(accepted, or out of rounds)──> END

Each function is a LangGraph node: it gets the state and returns only the keys it changes (graph.py wires them).
The answer is accepted only if: it parses as SolverAnswer, has 1-5 steps, every source URL was really returned by a
search in this run, and the judge LLM says it fits the cause. If anything fails for good, the supervisor's default steps stay.
"""
from __future__ import annotations

import json
from typing import Callable

from pydantic import ValidationError

from .llm import chat, complete_json, parse_json, resolve_llm, short_error
from .prompts import JUDGE_SYSTEM, SOLVER_SYSTEM, judge_prompt, solver_prompt
from .schemas import IncidentAssessment, Review, SolverAnswer
from .settings import Settings
from .tools import WEB_SEARCH_TOOL, web_search

MAX_STEPS = 5


def _close(state: dict, notes: list[str], answer: SolverAnswer | None = None) -> dict:
    """End the loop. With an accepted `answer` the assessment gets its solutions; otherwise it keeps the defaults."""
    assessment: IncidentAssessment = state["assessment"]
    update = {"notes": [*assessment.notes, *state.get("solver_notes", []), *notes]}
    if answer:
        update.update(summary=answer.summary, solutions=answer.solutions, verified=True, engine="llm")
    return {"assessment": assessment.model_copy(update=update), "done": True}


def check_answer(text: str, sources: dict[str, str]) -> tuple[SolverAnswer | None, list[str]]:
    """The rules that need no LLM. Returns (answer or None, problems)."""
    try:
        answer = parse_json(SolverAnswer, text)
    except ValidationError as e:
        return None, [f"the answer is not valid JSON of the required shape ({short_error(e, 120)})"]
    problems = []
    if not answer.summary.strip():
        problems.append("the summary is empty")
    if not 1 <= len(answer.solutions) <= MAX_STEPS or any(not s.step.strip() for s in answer.solutions):
        problems.append(f"give 1 to {MAX_STEPS} non-empty steps")
    for s in answer.solutions:
        if s.source and s.source not in sources:
            problems.append(f"source {s.source} was not returned by any search; use only URLs from search results or null")
    return answer, problems


def make_nodes(settings: Settings) -> tuple[Callable, Callable, Callable]:
    spec = resolve_llm(settings, "solver")
    tools = [WEB_SEARCH_TOOL] if settings.tavily_api_key else []

    def solve(state: dict) -> dict:
        """Ask the LLM what to do next: search, or give the final answer."""
        first = [] if state.get("messages") else [{"role": "system", "content": SOLVER_SYSTEM},
                                                  {"role": "user", "content": solver_prompt(state["assessment"])}]
        can_search = tools and len(state.get("queries", [])) < settings.solver_max_searches
        try:
            reply = chat(spec, [*state.get("messages", []), *first], tools if can_search else None)
        except Exception as e:
            return _close(state, [f"solver LLM failed ({short_error(e)}); default steps kept."])
        return {"messages": [*first, reply]}

    def run_tools(state: dict) -> dict:
        """Run the web searches the LLM asked for. A failed search is reported to the LLM as text, never raised."""
        results, queries, sources = [], [], {}
        for call in state["messages"][-1]["tool_calls"]:
            try:
                query = str(json.loads(call["function"]["arguments"])["query"])
                if len(state.get("queries", [])) + len(queries) >= settings.solver_max_searches:
                    content = "Search limit reached. Write the final answer now."
                else:
                    found = web_search(settings, query)
                    queries.append(query)
                    sources.update({r["url"]: r["title"] for r in found})
                    content = json.dumps(found)
            except Exception as e:
                content = f"search failed: {short_error(e)}"
            results.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"], "content": content})
        return {"messages": results, "queries": queries, "sources": sources}

    def verify(state: dict) -> dict:
        """Accept the answer or send it back with the reason."""
        answer, problems = check_answer(state["messages"][-1]["content"], state.get("sources", {}))
        if answer and not problems:
            try:
                review = complete_json(spec, JUDGE_SYSTEM, judge_prompt(state["assessment"], answer), Review)
            except Exception as e:
                return _close(state, [f"solver judge failed ({short_error(e)}); default steps kept."])
            if not review.ok:
                problems = [review.problem or "the judge rejected the answer"]
        if not problems:
            return _close(state, [], answer)
        attempt = state.get("rounds", 0) + 1
        note = f"solver answer {attempt} rejected: {'; '.join(problems)}"
        if attempt >= settings.solver_max_rounds:
            return _close(state, [note, "solver gave no acceptable answer; default steps kept."])
        feedback = {"role": "user", "content": f"Your answer was rejected: {'; '.join(problems)}. "
                                               "Fix it (search again if needed) and answer with the JSON only."}
        return {"messages": [feedback], "rounds": attempt, "solver_notes": [note]}

    return solve, run_tools, verify
