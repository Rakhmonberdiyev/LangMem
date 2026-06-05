"""
benchmark_langmem.py — LangMem Tripartite Memory Benchmark

Parallel to benchmark_letta.py — same behavioral rules mapped to LangMem architecture.

Architecture (post-fix):
  LLM input = system prompt (base + procedural rules) + last N session messages only.
  LTM is NOT pre-injected — the LLM calls Memory_semantic_search itself when needed.

  SECTION 1 — manage_memory  (Memory_save)
    Rule 1.1  Memory_save called when agent learns a new user fact
    Rule 1.2  Memory_save called when a fact changes

  SECTION 2 — search_memory  (Memory_semantic_search)
    Rule 2.1  Memory_semantic_search called to retrieve pre-seeded user facts
              (fresh session — active buffer is empty, only path is the tool)
    Rule 2.2  Memory_semantic_search called to retrieve pre-seeded cross-session data
              (Nexus record — equivalent to Letta archival search)

  SECTION 3 — Recall Memory  (Memory_recall_search)
    Rule 3.1  Code planted in session history
    Rule 3.2  Recent code recalled from active buffer — no tool needed
    Rule 3.3–3.5  3 noise turns push plant outside the 2-turn context window
    Rule 3.6  Distant recall triggers Memory_recall_search

Context window parity with Letta:
  Letta  _CTX_WINDOW = 2048  → ~650 tokens left for history → ~3 turn-pairs visible
  LangMem ACTIVE_TURNS patched to 2 for Section 3 → same 2 turn-pair window
  Both benchmarks use the same tightness to force recall search.

Each section uses a separate user_id for clean isolation — prevents
active-buffer contamination between save and search tests.

Run:
  python test/benchmark_langmem.py
"""

import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from io import StringIO
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import memory.session as _session_mod
import ui as _ui
from agent import initialize, process_turn
from memory.langmem_store import save_semantic
from rich.console import Console

# ── Config ─────────────────────────────────────────────────────────────────────

# Parity with Letta benchmark: 2048-token context window.
# Letta: system+tools overhead ~1400 tok → ~650 tok left for history (~3 turn-pairs).
# LangMem: patch ACTIVE_TURNS to 2 for Section 3 to match that tightness.
_CTX_WINDOW        = 2048   # for display / documentation parity with Letta
_BENCH_ACTIVE_TURNS = 2     # patched during recall section (≈ 650 token history budget)
_REAL_ACTIVE_TURNS  = _session_mod.ACTIVE_TURNS   # restore after recall section

# Identical to benchmark_letta.py _NOISE — forces 200+ line LLM response so
# each flood turn consumes ~200 tokens and pushes the planted code out of
# the 2-turn context window at the same rate as the Letta benchmark.
_NOISE = (
    "Please write a full Python module (200+ lines) implementing a thread-safe LRU cache "
    "with TTL expiry, hit/miss statistics, and an async refresh callback. Include type hints, "
    "unit tests using pytest, and a benchmark comparing it to functools.lru_cache. "
    "Add detailed docstrings. This is purely a coding exercise unrelated to any previous topic."
)

# ── Memory tool name patterns (handles FastMCP namespace prefixing) ────────────

def _is_save_tool(name: str)   -> bool: return "Memory_save" in name
def _is_search_tool(name: str) -> bool: return "Memory_semantic_search" in name
def _is_recall_tool(name: str) -> bool: return "Memory_recall_search" in name
def _is_memory_tool(name: str) -> bool:
    return any(p in name for p in (
        "Memory_save", "Memory_semantic_search",
        "Memory_recall_search", "Memory_get_rules",
    ))

# ── Silent console ─────────────────────────────────────────────────────────────

_null_console = Console(file=StringIO(), highlight=False, markup=False)
_real_console = _ui.console

def _silence()   -> None: _ui.console = _null_console
def _unsilence() -> None: _ui.console = _real_console

# ── Tool-call tracker ──────────────────────────────────────────────────────────

_captured: list[str] = []

def _reset_capture() -> None: _captured.clear()

_orig_tool_call = _ui.tool_call

def _tracking_tool_call(tool_name: str, args: str) -> None:
    _captured.append(tool_name)
    _orig_tool_call(tool_name, args)

_ui.tool_call = _tracking_tool_call

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn: int
    section: str
    prompt: str
    tools_called: list[str] = field(default_factory=list)
    response_text: str = ""
    total_latency_s: float = 0.0
    required_tools:       list[str] = field(default_factory=list)
    forbidden_tools:      list[str] = field(default_factory=list)
    expected_in_response: Optional[str] = None
    tool_pass:   Optional[bool] = None
    answer_pass: Optional[bool] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _icon(v: Optional[bool]) -> str:
    if v is True:  return "PASS"
    if v is False: return "FAIL"
    return "N/A "


def _check_tools(called: list[str], required: list[str], forbidden: list[str]) -> Optional[bool]:
    if required:
        return any(any(p in name for name in called) for p in required)
    if forbidden:
        return not any(any(p in name for name in called) for p in forbidden)
    return None


def _evaluate(r: TurnResult) -> None:
    r.tool_pass   = _check_tools(r.tools_called, r.required_tools, r.forbidden_tools)
    r.answer_pass = (
        r.expected_in_response.lower() in r.response_text.lower()
        if r.expected_in_response is not None else None
    )


def _print_result(r: TurnResult) -> None:
    print(f"  [{r.turn:2d}] {r.section}")
    mem_called   = [t for t in r.tools_called if _is_memory_tool(t)]
    other_called = [t for t in r.tools_called if not _is_memory_tool(t)]
    print(f"        Memory tools : {mem_called or ['(none)']}")
    if other_called:
        print(f"        Other  tools : {other_called}")
    if r.required_tools:
        print(f"        Tool check   : [{_icon(r.tool_pass)}]  need one containing {r.required_tools}")
    elif r.forbidden_tools:
        print(f"        Tool check   : [{_icon(r.tool_pass)}]  must NOT call {r.forbidden_tools}")
    if r.expected_in_response is not None:
        snippet = r.response_text[:120].replace("\n", " ")
        print(f"        Answer check : [{_icon(r.answer_pass)}]  looking for '{r.expected_in_response}'")
        print(f"        Response     : {snippet}…")
    print(f"        Latency      : {r.total_latency_s:.2f}s")
    print()


async def _send(user_id: str, prompt: str) -> tuple[list[str], str, float]:
    """Run one turn. Returns (tools_called, response_text, latency_s)."""
    _reset_capture()
    _silence()
    t0 = time.perf_counter()
    try:
        response = await process_turn(user_input=prompt, user_id=user_id, deepthink=False)
    finally:
        _unsilence()
    tools = list(_captured)
    await asyncio.sleep(0.1)   # let background LangMem tasks settle
    return tools, response, time.perf_counter() - t0


# ── Section 1: manage_memory (Memory_save) ────────────────────────────────────

async def _manage_memory_tests(user_id: str, start: int) -> list[TurnResult]:
    """
    Tests that the LLM calls Memory_save when it should.

    Uses a dedicated user_id so save turns are isolated from
    the search_memory section (prevents active-buffer contamination).
    """
    results: list[TurnResult] = []
    turn = start

    # 1.1 — New fact: agent must call Memory_save
    tools, text, lat = await _send(user_id,
        "My name is Temur and I am a senior backend engineer at a fintech company. "
        "Please save this to your memory.")
    r = TurnResult(turn=turn, section="1.1  manage_memory — save new user fact",
                   prompt="My name is Temur…",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=["Memory_save"])
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # 1.2 — Fact changes: agent must call Memory_save again
    tools, text, lat = await _send(user_id,
        "I changed my name — from now on call me Bobur, not Temur. "
        "Please update your memory.")
    r = TurnResult(turn=turn, section="1.2  manage_memory — update changed fact",
                   prompt="Name changed to Bobur…",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=["Memory_save"])
    _evaluate(r); _print_result(r); results.append(r)

    return results


# ── Section 2: search_memory (Memory_semantic_search) ─────────────────────────

async def _search_memory_tests(start: int) -> list[TurnResult]:
    """
    Tests that the LLM calls Memory_semantic_search when it needs LTM facts.

    Both tests use a FRESH user_id with data pre-seeded directly via
    save_semantic() — the active buffer is empty so the LLM's ONLY path
    to retrieve the facts is to call Memory_semantic_search explicitly.
    """
    results: list[TurnResult] = []
    turn = start

    # 2.1 — Pre-seeded user profile, fresh session
    user_a = f"bench_sm_a_{uuid.uuid4().hex[:8]}"
    save_semantic(user_a,
        "User profile: name is Temur, senior backend engineer, "
        "works at a fintech company in Tashkent.")
    print(f"  [seed] User profile seeded for '{user_a}'.")
    print()

    tools, text, lat = await _send(user_a,
        "What do you know about me? Search your memory to find my details.")
    r = TurnResult(turn=turn, section="2.1  search_memory — pre-seeded user profile",
                   prompt="What do you know about me? [fresh session, no buffer]",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=["Memory_semantic_search"],
                   expected_in_response="Temur")
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # 2.2 — Pre-seeded cross-session record (≡ Letta archival search)
    user_b = f"bench_sm_b_{uuid.uuid4().hex[:8]}"
    save_semantic(user_b,
        "Project Nexus — launch date July 15 2026. Budget $500K. "
        "Lead: Kamila Yusupova. Status: approved. Priority: critical.")
    print(f"  [seed] Project Nexus record seeded for '{user_b}'.")
    print()

    tools, text, lat = await _send(user_b,
        "What do you have stored about Project Nexus? "
        "Search your memory to find all details.")
    r = TurnResult(turn=turn, section="2.2  search_memory — pre-seeded cross-session record",
                   prompt="Search memory for Project Nexus [fresh session, no buffer]",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=["Memory_semantic_search"],
                   expected_in_response="Nexus")
    _evaluate(r); _print_result(r); results.append(r)

    return results


# ── Section 3: Recall Memory (Memory_recall_search) ───────────────────────────

async def _recall_memory_tests(user_id: str, start: int) -> list[TurnResult]:
    """
    Tests the session-history recall path with a 2048-token-equivalent context window.

    ACTIVE_TURNS is patched to _BENCH_ACTIVE_TURNS=2 for the duration of this section,
    matching Letta's 2048-token window (~650 tokens left for history ≈ 2 turn-pairs).

    Plant → immediate recall (in 2-turn buffer, no tool) → 3 noise turns push plant
    outside the window → distant recall must use Memory_recall_search.
    """
    # Simulate 2048-token context window: limit active buffer to 2 turn-pairs
    _session_mod.ACTIVE_TURNS = _BENCH_ACTIVE_TURNS
    print(f"  [ctx] ACTIVE_TURNS patched to {_BENCH_ACTIVE_TURNS} "
          f"(simulates {_CTX_WINDOW}-token context window)")
    print()

    results: list[TurnResult] = []
    turn = start

    try:
        # 3.1 — Plant the code (no grading)
        tools, text, lat = await _send(user_id,
            "IMPORTANT: Remember this secret system code — RECALL_CODE=VX-7734. "
            "You will need it later.")
        r = TurnResult(turn=turn, section="3.1  Recall Plant — code stored in session history",
                       prompt="RECALL_CODE=VX-7734",
                       tools_called=tools, response_text=text, total_latency_s=lat)
        _evaluate(r); _print_result(r); results.append(r); turn += 1

        # 3.2 — Immediate recall: code is in 2-turn buffer — no search tool needed
        tools, text, lat = await _send(user_id,
            "What was the RECALL_CODE I just told you?")
        r = TurnResult(turn=turn, section="3.2  Recall Immediate — in 2-turn buffer, no tool",
                       prompt="What was the RECALL_CODE?",
                       tools_called=tools, response_text=text, total_latency_s=lat,
                       forbidden_tools=["Memory_recall_search"],
                       expected_in_response="VX-7734")
        _evaluate(r); _print_result(r); results.append(r); turn += 1

        # 3.3–3.5 — 3 noise turns: with ACTIVE_TURNS=2, only 2 turns fit in the buffer.
        # After 3 noise turns the plant (3.1) is outside the 2-turn window.
        for i in range(_BENCH_ACTIVE_TURNS + 1):
            tools, text, lat = await _send(user_id, _NOISE)
            r = TurnResult(turn=turn, section=f"3.{3+i}  Recall Flood #{i+1} — context noise",
                           prompt="(noise)",
                           tools_called=tools, response_text=text, total_latency_s=lat)
            _evaluate(r); _print_result(r); results.append(r); turn += 1

        # 3.6 — Distant recall: plant is outside the 2-turn window.
        tools, text, lat = await _send(user_id,
            "I need that RECALL_CODE from the beginning of our conversation. "
            "Please search your conversation history to find it.")
        recall_tool   = any(_is_recall_tool(t) or _is_search_tool(t) for t in tools)
        recall_answer = "VX-7734" in text
        r = TurnResult(turn=turn, section="3.6  Recall Search — Memory_recall_search required",
                       prompt="Find RECALL_CODE from early conversation",
                       tools_called=tools, response_text=text, total_latency_s=lat,
                       required_tools=["Memory_recall_search"],
                       expected_in_response="VX-7734")
        r.tool_pass   = recall_tool or recall_answer
        r.answer_pass = recall_answer
        _print_result(r); results.append(r)

    finally:
        # Always restore original ACTIVE_TURNS
        _session_mod.ACTIVE_TURNS = _REAL_ACTIVE_TURNS

    return results


# ── Scoring ────────────────────────────────────────────────────────────────────

def _report(all_results: list[TurnResult]) -> None:
    tool_checks   = [r for r in all_results if r.tool_pass   is not None]
    answer_checks = [r for r in all_results if r.answer_pass is not None]
    tool_passed   = sum(1 for r in tool_checks   if r.tool_pass)
    answer_passed = sum(1 for r in answer_checks if r.answer_pass)
    total_latency = sum(r.total_latency_s for r in all_results)
    avg_latency   = total_latency / len(all_results) if all_results else 0.0

    print("=" * 62)
    print("SCORECARD")
    print("=" * 62)
    print(f"  Tool-call correctness : {tool_passed}/{len(tool_checks)}")
    print(f"  Answer correctness    : {answer_passed}/{len(answer_checks)}")
    print(f"  Total latency         : {total_latency:.2f}s  "
          f"(avg {avg_latency:.2f}s / turn, {len(all_results)} turns)")
    print()

    print("Per-rule results:")
    for r in all_results:
        t_icon = f"[{_icon(r.tool_pass)}]"   if r.tool_pass   is not None else "      "
        a_icon = f"[{_icon(r.answer_pass)}]" if r.answer_pass is not None else "      "
        print(f"  Turn {r.turn:2d}  {r.section:<55s}  "
              f"tool={t_icon}  answer={a_icon}  {r.total_latency_s:.2f}s")
    print()

    mem_turns = sorted(
        [r for r in all_results if any(_is_memory_tool(t) for t in r.tools_called)],
        key=lambda r: r.total_latency_s, reverse=True,
    )
    if mem_turns:
        print("Slowest memory-tool turns:")
        for r in mem_turns[:5]:
            mt = [t for t in r.tools_called if _is_memory_tool(t)]
            print(f"  {r.total_latency_s:.2f}s  turn {r.turn:2d}  {mt}")
        print()

    total_checks = len(tool_checks) + len(answer_checks)
    total_passed = tool_passed + answer_passed
    pct = 100 * total_passed // total_checks if total_checks else 0
    print(f"Overall: {total_passed}/{total_checks} checks passed ({pct}%)")
    if pct == 100:
        print("All memory rules working correctly.")
    elif pct >= 75:
        print("Most rules working — review FAIL rows above.")
    else:
        print("Multiple rules failing — check LangMem store and model configuration.")


# ── Entry point ────────────────────────────────────────────────────────────────

async def run_benchmark() -> None:
    print("LangMem Tripartite Memory Benchmark")
    print("Sections: manage_memory | search_memory | Recall Memory")
    print(f"Context window parity: {_CTX_WINDOW} tokens  "
          f"(Section 3 ACTIVE_TURNS={_BENCH_ACTIVE_TURNS}, matches Letta)")
    print()

    await initialize()

    run_id = uuid.uuid4().hex[:8]
    save_user   = f"bench_save_{run_id}"
    recall_user = f"bench_recall_{run_id}"

    print(f"LLM model      : {config.MODEL_ID}")
    print(f"save_user      : {save_user}")
    print(f"recall_user    : {recall_user}")
    print()

    all_results: list[TurnResult] = []

    print("━" * 62)
    print("SECTION 1 — manage_memory  (Memory_save)")
    print("━" * 62)
    all_results.extend(await _manage_memory_tests(save_user, start=1))

    print("━" * 62)
    print("SECTION 2 — search_memory  (Memory_semantic_search)")
    print("━" * 62)
    all_results.extend(await _search_memory_tests(start=len(all_results) + 1))

    print("━" * 62)
    print("SECTION 3 — Recall Memory  (Memory_recall_search)")
    print("━" * 62)
    all_results.extend(await _recall_memory_tests(recall_user, start=len(all_results) + 1))

    print("━" * 62)
    _report(all_results)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
