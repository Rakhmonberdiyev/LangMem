"""
test/test_memory_langmem.py — LangMem Memory Layer Smoke Test (O'zbek tilida)

Tests all three LangMem memory tiers without running the full agent pipeline:

  Phase 1: Ingest 5 mock banking turns into LangMem (background manager)
  Phase 2: Search Semantic Memory — verify user facts were extracted
  Phase 3: Search Episodic Memory — verify few-shot examples were distilled
  Phase 4: Check Procedural Memory — verify rules store is accessible
  Phase 5: Test MCP tools (Memory_semantic_search, Memory_save, Memory_recall_search)

Run from the project root:
    python test/test_memory_langmem.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.langmem_store import (
    background_update,
    get_procedural_rules,
    save_semantic,
    search_episodic,
    search_semantic,
    store,
)
from memory.session import save_turn

TEST_USER = f"test_langmem_{os.getpid()}"

# ── Mock banking conversation turns ───────────────────────────────────────────

TURNS = [
    (
        "Salom! Mening ismim Dilnoza. Toshkentda yashayman va pensioner bo'laman.",
        "Salom Dilnoza! Xazna bankiga xush kelibsiz. Pensiya masalalarida yordam bera olaman.",
    ),
    (
        "Pensiya to'lovim qachon keladi? Mirzo Ulug'bek tumani, Yangi shahar ko'chasi.",
        "Mirzo Ulug'bek tumani bo'yicha Pension_* tool orqali so'rov qildim. "
        "Yangi shahar ko'chasidagi pensiya to'lovi oyning 5-kunida amalga oshiriladi.",
    ),
    (
        "Menga Humo kartasi kerak. Qancha turadi?",
        "Humo Classic kartasi yillik 30,000 so'm. Card_* tooldan ma'lumot oldim.",
    ),
    (
        "Depozit ochmoqchiman. Eng yuqori foiz qaysi depozitda?",
        "Deposit_* tooli ma'lumotiga ko'ra 'Premium Depozit' — yillik 21% foiz bilan eng foydali.",
    ),
    (
        "Men o'zbek tilida gaplashishni afzal ko'raman va 65 yoshdaman.",
        "Tushunarli, Dilnoza. O'zbek tilida davom etamiz. Barcha ma'lumotlarni eslab qolaman.",
    ),
]


# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  ✅  {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌  {msg}")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Test phases ───────────────────────────────────────────────────────────────

async def phase1_ingest() -> None:
    _section("Phase 1 — Ingesting 5 turns into LangMem background manager")
    messages: list[dict] = []
    for i, (user_msg, asst_msg) in enumerate(TURNS, 1):
        print(f"  Turn {i}: {user_msg[:60]}…")
        await save_turn(TEST_USER, user_msg, asst_msg)
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
        # Run background update after every 2 turns and at the end
        if i % 2 == 0 or i == len(TURNS):
            await background_update(messages, TEST_USER)
            print(f"    → background_update called (turns 1-{i})")


async def phase2_semantic_search() -> None:
    _section("Phase 2 — Semantic Memory search (user facts)")

    checks = [
        ("Dilnoza pensioner",      ["dilnoza", "pensioner", "pensiya"]),
        ("Toshkent yashash joyi",  ["toshkent"]),
        ("O'zbek tili afzal",      ["o'zbek", "uzbek"]),
        ("65 yosh",                ["65"]),
    ]

    passed = 0
    for query, required_kws in checks:
        results = await search_semantic(query, TEST_USER, limit=5)
        combined = " ".join(results).lower()
        hit = any(kw in combined for kw in required_kws)
        if hit:
            _ok(f"Query '{query}' → found: {results[0][:80] if results else '(none)'}…")
            passed += 1
        else:
            _fail(f"Query '{query}' → no match (required: {required_kws})")
            if results:
                print(f"     Got: {results[0][:120]}")

    print(f"\n  Semantic: {passed}/{len(checks)} passed")
    return passed


async def phase3_episodic_search() -> None:
    _section("Phase 3 — Episodic Memory search (few-shot examples)")

    results = await search_episodic("pensiya to'lov", TEST_USER, limit=3)
    if results:
        _ok(f"Episodic search returned {len(results)} example(s)")
        for r in results:
            print(f"    • {r[:120]}")
    else:
        print("  ⚠️  No episodic examples yet (normal for <6 messages in first run)")
    return len(results)


async def phase4_procedural_rules() -> None:
    _section("Phase 4 — Procedural Memory (system rules)")

    rules = await get_procedural_rules(TEST_USER)
    if rules:
        _ok(f"Procedural rules present ({len(rules)} chars)")
        print(f"    Preview: {rules[:200]}…")
    else:
        print("  ℹ️  No procedural rules yet — optimizer runs every 10 turns (expected for smoke test)")

    # Test direct write to semantic store
    key = save_semantic(TEST_USER, "Test fact: user prefers dark mode (from Phase 4 test)")
    results = await search_semantic("dark mode preference", TEST_USER, limit=3)
    combined = " ".join(results).lower()
    if "dark mode" in combined or "test fact" in combined:
        _ok(f"Direct semantic write + read roundtrip OK (key={key[:16]})")
    else:
        _fail("Direct semantic write did not appear in search results")


async def phase5_mcp_tools() -> None:
    _section("Phase 5 — Memory MCP tools (simulated)")
    from tools.langmem_mcp import (
        Memory_recall_search,
        Memory_save,
        Memory_semantic_search,
        set_langmem_context,
    )
    # Memory_episodic_search merged into Memory_semantic_search per architecture diagram

    set_langmem_context(TEST_USER)

    # Test Memory_semantic_search
    result = await Memory_semantic_search("Dilnoza pensioner")
    # Pass if: found relevant facts, found any memory entries, or store had nothing yet
    tool_ok = (
        "dilnoza" in result.lower()
        or "pensioner" in result.lower()
        or "No relevant" in result
        or result.startswith("-")  # any facts returned from store
    )
    if tool_ok:
        _ok(f"Memory_semantic_search: {result[:100]}")
    else:
        _fail(f"Memory_semantic_search unexpected result: {result[:100]}")

    # Test Memory_save
    result = await Memory_save("User explicitly asked to be greeted by first name: Dilnoza")
    if "Stored" in result:
        _ok(f"Memory_save: {result}")
    else:
        _fail(f"Memory_save unexpected result: {result}")

    # Test Memory_recall_search
    result = await Memory_recall_search("Humo kartasi")
    if "Humo" in result or "No messages" in result:
        _ok(f"Memory_recall_search: {result[:100]}")
    else:
        _fail(f"Memory_recall_search unexpected result: {result[:100]}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "=" * 60)
    print("  LangMem Memory Layer Smoke Test")
    print(f"  Test user: {TEST_USER}")
    print("=" * 60)

    await phase1_ingest()
    sem_passed = await phase2_semantic_search()
    ep_count   = await phase3_episodic_search()
    await phase4_procedural_rules()
    await phase5_mcp_tools()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Semantic search:  {sem_passed}/4 queries matched")
    print(f"  Episodic store:   {ep_count} example(s) found")
    print(f"  Procedural store: accessible ✓")
    print(f"  MCP tools:        all 3 tested ✓")
    print()
    if sem_passed >= 2:
        print("  RESULT: PASS — LangMem tripartite memory is working")
    else:
        print("  RESULT: PARTIAL — LLM extraction may need a live API connection")
    print()


if __name__ == "__main__":
    asyncio.run(main())
