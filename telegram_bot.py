"""
Telegram bot interface — like Gemini's chat UI.

Features:
  • Mode toggle button: 🧠 Deepthink  ↔  ⚡ Fast
  • New Session button (clears Redis history, keeps Mem0 LTM)
  • "Processing…" indicator while the agent is thinking
  • Per-user mode stored in memory
  • Each Telegram user_id maps directly to the agent's user_id

"""

import os
import asyncio
import logging
import html as _html
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
)
from aiogram.filters import CommandStart, Command
from aiogram.enums import ChatAction
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest

load_dotenv()
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for _noisy in ("aiogram", "aiohttp", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import config
from agent import process_turn, initialize
from memory.session import clear_session
from tools.ingestion import ingest_document
from memory.session import add_user_doc
import ui

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ── Per-user settings (in-memory) ─────────────────────────────────────────────
# {telegram_user_id: {"deepthink": bool}}
_settings: dict[int, dict] = {}


def _get_deepthink(uid: int) -> bool:
    return _settings.setdefault(uid, {"deepthink": True})["deepthink"]


def _set_deepthink(uid: int, val: bool) -> None:
    _settings.setdefault(uid, {})["deepthink"] = val


# ── Inline keyboard ────────────────────────────────────────────────────────────

def _keyboard(uid: int) -> InlineKeyboardMarkup:
    deepthink = _get_deepthink(uid)
    toggle_label = (
        "⚡ Switch to Fast"
        if deepthink
        else "🧠 Switch to Deepthink"
    )
    mode_label = "🧠 Deepthink ✓" if deepthink else "⚡ Fast ✓"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=mode_label,    callback_data="noop"),
                InlineKeyboardButton(text=toggle_label,  callback_data="toggle_mode"),
            ],
            [
                InlineKeyboardButton(text="🆕 New Session", callback_data="new_session"),
            ],
        ]
    )


# ── Message helpers ────────────────────────────────────────────────────────────

def _split_4096(text: str) -> list[str]:
    """Split at 4096-char Telegram limit on word boundaries."""
    MAX = 4000
    if len(text) <= MAX:
        return [text]
    chunks, buf = [], ""
    for word in text.split(" "):
        if len(buf) + len(word) + 1 > MAX:
            chunks.append(buf.rstrip())
            buf = word + " "
        else:
            buf += word + " "
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# ── Typing indicator loop ──────────────────────────────────────────────────────

async def _keep_typing(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    """Re-send ChatAction.TYPING every 4 s until stop is set (Telegram expires it after ~5 s)."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(stop.wait()), timeout=4.0)
        except asyncio.TimeoutError:
            pass


# ── Handlers ──────────────────────────────────────────────────────────────────

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    uid = message.from_user.id
    _set_deepthink(uid, True)
    name = message.from_user.first_name or "there"
    await message.answer(
        f"👋 Hi <b>{name}</b>! I'm your AI Assistant.\n\n"
        "🧠 <b>Deepthink</b> — I reason step-by-step, search for evidence, "
        "then synthesize a careful answer.\n"
        "⚡ <b>Fast</b> — Direct response with tool access, no deep reasoning.\n\n"
        "Just send me a message to get started!",
        reply_markup=_keyboard(uid),
    )


@router.message(Command("mode"))
async def cmd_mode(message: Message) -> None:
    uid = message.from_user.id
    mode = "🧠 Deepthink" if _get_deepthink(uid) else "⚡ Fast"
    await message.answer(f"Current mode: <b>{mode}</b>", reply_markup=_keyboard(uid))


@router.message(Command("newsession"))
async def cmd_newsession(message: Message) -> None:
    uid = message.from_user.id
    await clear_session(str(uid))
    await message.answer(
        "🆕 <b>New session started!</b>\n"
        "Conversation history cleared. Long-term memories are preserved.",
        reply_markup=_keyboard(uid),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "<b>Commands:</b>\n"
        "/start      — Welcome + reset mode\n"
        "/mode       — Show current mode\n"
        "/newsession — Clear conversation history\n"
        "/help       — This message\n\n"
        "<b>Buttons on each reply:</b>\n"
        "🧠 Deepthink / ⚡ Fast  — switch reasoning mode\n"
        "🆕 New Session          — start fresh conversation\n\n"
        "<b>Long-term memory</b> is always active — I remember facts "
        "about you across sessions."
    )


@router.callback_query(F.data == "noop")
async def cb_noop(cb: CallbackQuery) -> None:
    await cb.answer()


@router.callback_query(F.data == "toggle_mode")
async def cb_toggle(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    new_val = not _get_deepthink(uid)
    _set_deepthink(uid, new_val)
    label = "🧠 Deepthink" if new_val else "⚡ Fast"
    await cb.answer(f"Switched to {label} mode")
    try:
        await cb.message.edit_reply_markup(reply_markup=_keyboard(uid))
    except TelegramBadRequest:
        pass


@router.callback_query(F.data == "new_session")
async def cb_new_session(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    await clear_session(str(uid))
    await cb.answer("Session cleared!")
    await cb.message.answer(
        "🆕 <b>New session started!</b>\n"
        "Previous conversation cleared. Long-term memories preserved.",
        reply_markup=_keyboard(uid),
    )


@router.message(F.document)
async def handle_document(message: Message) -> None:
    uid  = message.from_user.id
    doc  = message.document
    name = doc.file_name or "document"
    ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext not in ("pdf", "txt", "md"):
        await message.answer(
            "⚠️ Unsupported file type. Please send a <b>PDF</b>, <b>TXT</b>, or <b>MD</b> file.",
            reply_markup=_keyboard(uid),
        )
        return

    status = await message.answer(f"📄 <i>Indexing <b>{name}</b>…</i>")
    ui.section(f"Document ingestion — {name}  (user {uid})")

    try:
        file = await message.bot.get_file(doc.file_id)
        buf  = await message.bot.download_file(file.file_path)
        data = buf.read()

        n_chunks = await asyncio.to_thread(ingest_document, data, name, str(uid))
        await add_user_doc(str(uid), name)

        ui.ok(f"Indexed {n_chunks} chunks from '{name}'")
        await status.edit_text(
            f"✅ <b>{name}</b> indexed — {n_chunks} chunks added to the knowledge base.\n"
            "You can now ask me questions about this document.",
            reply_markup=_keyboard(uid),
        )
    except Exception as exc:
        ui.err(f"Ingestion error: {exc}")
        safe = _html.escape(str(exc))
        await status.edit_text(
            f"❌ <b>Ingestion failed:</b> {safe}",
            reply_markup=_keyboard(uid),
        )


@router.message(F.text)
async def handle_message(message: Message) -> None:
    uid  = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        return

    deepthink   = _get_deepthink(uid)
    mode_emoji  = "🧠" if deepthink else "⚡"
    mode_name   = "Deepthink" if deepthink else "Fast"

    ui.section(f"Telegram → user {uid}  {mode_emoji} {mode_name}")

    # Start continuous typing indicator + status message
    _stop_typing = asyncio.Event()
    _typing_task = asyncio.create_task(
        _keep_typing(message.bot, message.chat.id, _stop_typing)
    )
    status = await message.answer(
        f"<i>{mode_emoji} {mode_name} mode — processing…</i>"
    )

    try:
        response = await process_turn(text, str(uid), deepthink=deepthink)

        await status.delete()

        chunks = _split_4096(response)
        for i, chunk in enumerate(chunks):
            kb = _keyboard(uid) if i == len(chunks) - 1 else None
            try:
                await message.answer(chunk, reply_markup=kb)
            except TelegramBadRequest:
                await message.answer(_html.escape(chunk), reply_markup=kb)

    except Exception as exc:
        logging.exception("process_turn error")
        safe = _html.escape(str(exc))[:300]
        try:
            await status.edit_text(
                f"❌ <b>Error:</b> {safe}",
                reply_markup=_keyboard(uid),
            )
        except TelegramBadRequest:
            await message.answer(f"❌ Error: {safe}", reply_markup=_keyboard(uid))

    finally:
        _stop_typing.set()
        await asyncio.gather(_typing_task, return_exceptions=True)


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "Add it to .env:  TELEGRAM_BOT_TOKEN=<your-token>"
        )

    await initialize()

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    await bot.set_my_commands([
        BotCommand(command="start",      description="Start / reset"),
        BotCommand(command="mode",       description="Show current mode"),
        BotCommand(command="newsession", description="Clear conversation history"),
        BotCommand(command="help",       description="Help & commands"),
    ])

    dp = Dispatcher()
    dp.include_router(router)

    from rich.panel import Panel
    ui.console.print(Panel(
        "[bold green]Telegram bot is live — waiting for messages[/bold green]\n"
        "[dim]Terminal shows the full pipeline for every incoming message.\n"
        "Chat happens in Telegram. No input needed here.[/dim]",
        border_style="green",
        padding=(0, 2),
    ))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
