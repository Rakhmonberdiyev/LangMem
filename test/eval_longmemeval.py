"""
test/eval_longmemeval.py — Mini-LongMemEval: Bank Tizimi Vaqtinchalik Bilimlar Benchmark testi

30 simulyatsiya qilingan kun davomida ketma-ket sodir bo'lgan 20 ta bank voqealari.
Ikkita asosiy baholash o'qi:
  A) Bilimlarning yangilanishi (Knowledge Updates) — Tizim HOZIRGI holatni to'g'ri saqlaydimi?
  B) Vaqtinchalik mantiq (Temporal Reasoning)    — Tizim o'zgarishlar QACHON bo'lganini aniqlay oladimi?

Baholash:  O'TDI = 2 ball  |  QISMAN = 1 ball  |  YIQILDI = 0 ball  |  maksimal = 20 ball

Tekshirilayotgan arxitektura:
  LangMem InMemoryStore (Semantic + Episodic tiers)
  Semantic manager : fakt ekstraktsiya + deduplication
  Episodic manager : few-shot misol distillatsiyasi

Ishga tushirish:
  python test/eval_longmemeval.py               # To'liq ishga tushirish (ma'lumot yuklash + baholash)
  python test/eval_longmemeval.py --skip-ingest # Mavjud foydalanuvchi ma'lumotlarini qayta baholash
"""

import asyncio
import os
import sys
import uuid
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.ltm import upsert_ltm, search_ltm

# ─────────────────────────────────────────────────────────────────────────────
# 1.  20 TA BANK VOQEALARI VAQTLAR CHIZIG'I (30 simulyatsiya qilingan kun)
# ─────────────────────────────────────────────────────────────────────────────

TIMELINE: list[tuple[int, str, str]] = [
    (1,
     "1-kun: Men Humo kartasini ochmoqchiman",
     "Ajoyib! Men sizning Humo kartasi uchun arizangizni boshladim. "
     "Sizga pasport va daromadni tasdiqlovchi hujjat kerak bo'ladi."),

    (2,
     "2-kun: Humo Classic kartasining yillik xizmat haqi qancha?",
     "Humo Classic birinchi yil uchun bepul, keyingi yillardan boshlab yiliga 50,000 UZS."),

    (3,
     "3-kun: Mening oylik daromadim 5,000,000 UZS",
     "Sizning oyiga 5,000,000 UZS daromadingiz Humo Classic yoki Premium kartalariga mos keladi."),

    (4,
     "4-kun: Men aynan Humo Classic darajasini afzal ko'raman",
     "Tushunarlidir. Arizangiz Humo Classic kartasi uchun tasdiqlandi."),

    (5,
     "5-kun: Humo Classic kontaktsiz to'lovlarni qo'llab-quvvatlaydimi?",
     "Ha, Humo Classic HumoPass texnologiyasi orqali kontaktsiz to'lovlarni qo'llab-quvvatlaydi."),

    (6,
     "6-kun: Mening billing manzilim: Toshkent shahri, Amir Temur ko'chasi, 12-uy",
     "Billing manzili yozib olindi: Toshkent shahri, Amir Temur ko'chasi, 12-uy."),

    (7,
     "7-kun: Men karta ko'chirmalarini o'zbek tilida olishni xohlayman",
     "Hisobingizda ko'chirma tili afzalligi o'zbek tiliga o'rnatildi."),

    (8,
     "8-kun: Humo Classic uchun kunlik naqd pul yechish limiti qancha?",
     "Humo Classic kartasidan kuniga 2,000,000 UZS gacha naqd pul yechish mumkin."),

    (9,
     "9-kun: Hisobimda SMS-xabarnomalarni yoqing",
     "Hisobingizda SMS-xabarnomalar muvaffaqiyatli yoqildi."),

    # ── ASOSIY O'ZGARISH: Humo → Visa ─────────────────────────────────────────
    (10,
     "10-kun: Humo arizamni bekor qiling — o'rniga menga Visa karta bering",
     "Tushundim. Sizning Humo Classic arizangiz bekor qilindi. "
     "Siz uchun yangi Visa karta arizasini boshlayapman."),

    (11,
     "11-kun: Men aynan Visa Gold kartasini xohlayman",
     "Visa Gold tasdiqlandi. Oyiga 5,000,000 UZS daromadingiz ushbu kartaga mos keladi "
     "(minimal talab 3,000,000 UZS). Ariza yangilandi."),

    (12,
     "12-kun: Visa Gold qanday sayohat imtiyozlarini taklif qiladi?",
     "Visa Gold quyidagilarni o'z ichiga oladi: xalqaro sayohat sug'urtasi, Priority Pass aeroport "
     "biznes-zallariga kirish va xorijiy valyutadagi tranzaksiyalar uchun komissiyasiz xizmat."),

    (14,
     "14-kun: Men tez-tez xizmat safari bilan sayohat qilaman, shuning uchun Visa Gold to'g'ri tanlov",
     "Tushundim. Sayohat qiluvchilar uchun Visa Gold ideal. Arizangiz Visa Gold uchun tasdiqlandi."),

    (15,
     "15-kun: Sobiq arizaga rafiqam Nilufarni Visa Gold kartasiga hammuallif (co-holder) sifatida qo'shing",
     "Hammuallif Nilufar sizning Visa Gold karta arizangizga qo'shildi."),

    (16,
     "16-kun: Mening kredit limitimni 10,000,000 UZS qilib belgilang",
     "Sizning Visa Gold kartangiz uchun kredit limiti 10,000,000 UZS qilib belgilandi."),

    (18,
     "18-kun: Visa Gold kartamda xorijiy valyuta tranzaksiyalarini yoqing",
     "Sizning Visa Gold kartangizda xorijiy valyutadagi tranzaksiyalar yoqildi."),

    # ── ASOSIY O'ZGARISH: Hammuallif olib tashlandi ───────────────────────────
    (20,
     "20-kun: Hammuallif qo'shish so'rovini bekor qiling — kartani faqat mening nomimda qoldiring",
     "Hammuallif Nilufar olib tashlandi. Visa Gold kartangiz faqat sizning "
     "nomingizga yagona egasi (sole-holder) sifatida chiqariladi."),

    # ── ASOSIY O'ZGARISH: Kredit limiti kamaytirildi ───────────────────────────
    (22,
     "22-kun: Kredit limitimni 10,000,000 dan 7,000,000 UZS ga kamaytiring",
     "Visa Gold kartangizdagi kredit limiti 10,000,000 UZS dan 7,000,000 UZS ga yangilandi."),

    (25,
     "25-kun: Toshkent elektr energiyasi to'lovlari uchun avtomatik to'lovni sozlang",
     "Sizning Visa Gold kartangiz orqali Toshkent elektr to'lovlari uchun avtomat to'lov sozlandi."),

    (28,
     "28-kun: Mening Visa Gold arizamning hozirgi holati qanday?",
     "Sizning Visa Gold arizangiz — yagona egalik, 7,000,000 UZS limit, "
     "xorijiy valyuta yoqilgan — ko'rib chiqilmoqda. 3 ish kunida tasdiqlanishi kutilmoqda."),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BAHOLASH SAVOLLARI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalQ:
    qid: str
    category: Literal["knowledge_update", "temporal_reasoning"]
    question: str
    expected_answer: str
    required_kws: list[str]
    stale_kws: list[str]


EVAL_QUESTIONS: list[EvalQ] = [
    # ── BILIMLARNING YANGILANISHI (5) ──────────────────────────────────────────
    EvalQ(
        qid="BY-1",
        category="knowledge_update",
        question="Foydalanuvchi hozirda qaysi kartani xohlamoqda yoki qaysi biriga ariza topshirgan?",
        expected_answer="Visa Gold (10-kunda Humo Classic'dan o'zgartirildi)",
        required_kws=["visa", "gold"],
        stale_kws=["humo"],
    ),
    EvalQ(
        qid="BY-2",
        category="knowledge_update",
        question="Hozirgi vaqtda foydalanuvchining kartasida hammuallif (co-holder) bormi?",
        expected_answer="Yo'q — hammuallif Nilufar 20-kunda bekor qilingan; karta faqat bitta egasiga tegishli",
        required_kws=["yagona", "bekor", "nilufar", "olib", "nomimda"],
        stale_kws=[],
    ),
    EvalQ(
        qid="BY-3",
        category="knowledge_update",
        question="Foydalanuvchining hozirgi kredit limiti qancha?",
        expected_answer="7,000,000 UZS (22-kunda 10 mlndan kamaytirilgan)",
        required_kws=["7,000,000", "7000000", "7 mln", "7 million"],
        stale_kws=["10,000,000", "10 mln", "10 million"],
    ),
    EvalQ(
        qid="BY-4",
        category="knowledge_update",
        question="Foydalanuvchi karta ko'chirmalari uchun qaysi tilni afzal ko'radi?",
        expected_answer="O'zbek tili (7-kunda sozlangan)",
        required_kws=["o'zbek", "uzbek"],
        stale_kws=[],
    ),
    EvalQ(
        qid="BY-5",
        category="knowledge_update",
        question="Foydalanuvchi uchun xorijiy valyutadagi tranzaksiyalar yoqilganmi?",
        expected_answer="Ha — 18-kunda yoqilgan",
        required_kws=["xorijiy", "valyuta", "yoqilgan", "tranzaksiya"],
        stale_kws=[],
    ),
    # ── VAQTINCHALIK MANTIQ (5) ────────────────────────────────────────────────
    EvalQ(
        qid="VM-1",
        category="temporal_reasoning",
        question="Foydalanuvchi boshqa kartaga o'tishdan oldin dastlab qaysi kartaga ariza topshirgan edi?",
        expected_answer="Humo Classic (1-9 kunlar), keyin 10-kunda Visa Gold'ga o'tgan",
        required_kws=["humo", "classic"],
        stale_kws=[],
    ),
    EvalQ(
        qid="VM-2",
        category="temporal_reasoning",
        question="Foydalanuvchining kredit limiti kamaytirilishidan oldin qancha edi?",
        expected_answer="10,000,000 UZS (16-kunda o'rnatilib, 22-kunda 7 mlnga tushirilgan)",
        required_kws=["10,000,000", "10 mln", "10 million", "10000000"],
        stale_kws=[],
    ),
    EvalQ(
        qid="VM-3",
        category="temporal_reasoning",
        question="Foydalanuvchining billing manzili nima?",
        expected_answer="Amir Temur ko'chasi, 12-uy, Toshkent (6-kunda sozlangan, o'zgarmagan)",
        required_kws=["amir temur", "toshkent", "12"],
        stale_kws=[],
    ),
    EvalQ(
        qid="VM-4",
        category="temporal_reasoning",
        question="Ushbu kartaga qachondir hammuallif qo'shish so'rovi bo'lganmidi?",
        expected_answer="Ha — Nilufar 15-kunda qo'shilgan, keyin 20-kunda olib tashlangan",
        required_kws=["nilufar", "hammuallif", "co-holder"],
        stale_kws=[],
    ),
    EvalQ(
        qid="VM-5",
        category="temporal_reasoning",
        question="Foydalanuvchi qancha oylik daromad ma'lum qilgan edi?",
        expected_answer="oyiga 5,000,000 UZS (3-kunda e'lon qilingan)",
        required_kws=["5,000,000", "5 mln", "5000000", "5 million"],
        stale_kws=[],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BAHOLASH MANTIQI
# ─────────────────────────────────────────────────────────────────────────────

def _kw_match(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def score_question(q: EvalQ, retrieved_facts: str) -> tuple[int, str]:
    """0 = YIQILDI | 1 = QISMAN | 2 = O'TDI"""
    facts = retrieved_facts.lower()

    found_required = _kw_match(facts, q.required_kws)
    found_stale    = _kw_match(facts, q.stale_kws) if q.stale_kws else False

    if q.category == "knowledge_update":
        if found_required and not found_stale:
            return 2, "O'TDI — joriy fakt mavjud, eski ma'lumot aralashmagan"
        if found_required and found_stale:
            return 1, "QISMAN — joriy fakt topildi, lekin eski fakt ham qaytarildi"
        return 0, "YIQILDI — kutilgan joriy fakt LTMdan topilmadi"
    else:  # temporal_reasoning
        if found_required:
            return 2, "O'TDI — tarixiy fakt LTM xotirasidan muvaffaqiyatli yuklandi"
        return 0, "YIQILDI — tarixiy fakt LTM xotirasida mavjud emas"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ASOSIY PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "═" * 64
SEP2 = "─" * 64

def _h1(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

GRADE_LABELS = {2: "✅ O'TDI   ", 1: "⚠️  QISMAN ", 0: "❌ YIQILDI "}


async def ingest_timeline(user_id: str) -> None:
    _h1("1-FAZA — 20 TA BANK VOQEASINI TIZIMGA YUKLASH")
    print(f"  Foydalanuvchi ID : {user_id}")
    print(f"  Voqealar soni    : 30 simulyatsiya kunida {len(TIMELINE)} ta\n")

    # Cumulative history so episodic manager gets enough context (needs ≥6 msgs)
    cumulative: list[dict] = []

    for i, (day, user_msg, asst_msg) in enumerate(TIMELINE, 1):
        sys.stdout.write(f"  [{i:02d}/20] {day:>2d}-kun: {user_msg[:55]}...")
        sys.stdout.flush()
        t0 = time.perf_counter()

        cumulative.append({"role": "user",      "content": user_msg})
        cumulative.append({"role": "assistant", "content": asst_msg})

        await upsert_ltm(
            user_msg,
            asst_msg,
            user_id,
            full_history=cumulative,
        )
        elapsed = time.perf_counter() - t0
        print(f"  ({elapsed:.1f}s)")
        await asyncio.sleep(0.2)

    print(f"\n  ✓ Ma'lumotlarni yuklash yakunlandi.")


async def evaluate(user_id: str) -> list[dict]:
    _h1("2-FAZA — BAHOLASH  (10 ta savol)")
    print(f"  Foydalanuvchi uchun LTM so'rovi yuborilmoqda: {user_id}\n")

    results = []
    for q in EVAL_QUESTIONS:
        print(f"\n  [{q.qid}] {q.question}")
        facts = await search_ltm(q.question, user_id, limit=7)
        score, reason = score_question(q, facts)
        grade = GRADE_LABELS[score]
        print(f"         Natija     : {grade}  ({score}/2)")
        print(f"         Kutilgan   : {q.expected_answer}")
        print(f"         Izoh       : {reason}")
        if facts:
            for line in facts.splitlines()[:3]:
                print(f"         LTM fakt   : {line[:90]}")
        results.append({
            "qid": q.qid,
            "category": q.category,
            "question": q.question,
            "expected": q.expected_answer,
            "score": score,
            "reason": reason,
            "retrieved_facts": facts,
        })

    return results


def print_scorecard(results: list[dict]) -> None:
    _h1("NATIJALAR JADVALI (SCORECARD)")

    ku = [r for r in results if r["category"] == "knowledge_update"]
    tr = [r for r in results if r["category"] == "temporal_reasoning"]

    def _section(title: str, rows: list[dict]) -> None:
        total = sum(r["score"] for r in rows)
        max_pts = len(rows) * 2
        print(f"\n  {title}  ({total}/{max_pts} ball)")
        print(f"  {'Savol ID':<10} {'Ball':<8} {'Xulosa'}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*40}")
        for r in rows:
            grade = GRADE_LABELS[r["score"]]
            print(f"  {r['qid']:<10} {r['score']}/2     {grade}  {r['reason']}")

    _section("A) Bilimlarning yangilanishi (Knowledge Updates)", ku)
    _section("B) Vaqtinchalik mantiq (Temporal Reasoning)", tr)

    grand = sum(r["score"] for r in results)
    max_g = len(results) * 2
    pct   = grand / max_g * 100

    print(f"\n{SEP2}")
    print(f"  YAKUNIY BALL : {grand} / {max_g} ball  ({pct:.0f}%)")
    print(SEP2)

    ku_score = sum(r["score"] for r in ku)
    tr_score = sum(r["score"] for r in tr)

    print(f"""
  RAHBARIYAT UCHUN HISOBOT
  ──────────────────────────────────────
  Bilimlarning yangilanishi : {ku_score}/{len(ku)*2} ball
  Vaqtinchalik mantiq       : {tr_score}/{len(tr)*2} ball

  LangMem arxitekturasi:
    • Semantic manager  — foydalanuvchi faktlarini extrakt qilib, deduplication qiladi
    • Episodic manager  — sifatli interaksiyalarni few-shot misol sifatida saqlaydi
    • enable_deletes=False — tarixiy faktlar o'chirilmaydi (temporal reasoning uchun)

  Xulosa: {pct:.0f}% aniqlik
""")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  KIRISH NUQTASI
# ─────────────────────────────────────────────────────────────────────────────

async def main(skip_ingest: bool, user_id: str) -> None:
    print(f"\n{'═'*64}")
    print("  MINI-LongMemEval — Bank Tizimi Vaqtinchalik Bilimlar Benchmark")
    print(f"{'═'*64}")
    print(f"  Foydalanuvchi ID : {user_id}")
    print(f"  Voqealar soni    : {len(TIMELINE)} | Savollar : {len(EVAL_QUESTIONS)}")
    mode_label = "faqat baholash (--skip-ingest)" if skip_ingest else "to'liq ijro"
    print(f"  Rejim            : {mode_label}")

    if not skip_ingest:
        await ingest_timeline(user_id)
        print("\n  LangMem background extraction yakunlanishi uchun 3 soniya kutilmoqda…")
        await asyncio.sleep(3)
    else:
        print(f"\n  Yuklash tashlab ketildi — {user_id} uchun mavjud ma'lumotlar ishlatilmoqda")

    results = await evaluate(user_id)
    print_scorecard(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-LongMemEval banking benchmark")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Yuklash bosqichini o'tkazib yuborish")
    parser.add_argument("--user-id", default=f"eval_{uuid.uuid4().hex[:8]}",
                        help="Foydalanuvchi ID (standart: tasodifiy)")
    args = parser.parse_args()
    asyncio.run(main(skip_ingest=args.skip_ingest, user_id=args.user_id))
