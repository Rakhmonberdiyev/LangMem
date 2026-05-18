"""
Telegram bot interface — like Gemini's chat UI.

Features:
  • Mode toggle button: 🧠 Deepthink  ↔  ⚡ Fast
  • Multi-session support: /sessions to view & switch, ➕ New Session to branch
  • Continuous typing indicator while the agent is thinking
  • Per-user mode stored in memory
  • Each Telegram user_id maps directly to the agent's user_id
  • Document ingestion: PDF, TXT, MD → Qdrant RAG

"""

import os
import asyncio
import logging
import html as _html
import time as _time
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
from memory.session import (
    clear_session, add_user_doc,
    create_session, switch_session,
    get_sessions_list, get_current_session_id,
)
from tools.ingestion import ingest_document
import ui

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


# ── Streaming writer ───────────────────────────────────────────────────────────

class _TelegramStreamer:
    """Accumulates LLM text chunks and periodically edits a Telegram message.

    Debouncing: edits at most once every MIN_SECS seconds OR every MIN_CHARS
    new characters, whichever comes first. Shows a blinking cursor while
    streaming; finalize() removes it and returns the full buffered text.
    """
    CURSOR   = "▌"
    MIN_CHARS = 60    # min new chars before forcing an edit
    MIN_SECS  = 2.0   # min seconds between edits

    def __init__(self, status_message, prefix: str = ""):
        self._msg        = status_message
        self._prefix     = prefix   # plain-text header shown above the response
        self._buf        = ""
        self._last_len   = 0        # len(_buf) at last Telegram edit
        self._last_ts    = 0.0
        self._lock       = asyncio.Lock()

    async def __call__(self, chunk: str) -> None:
        async with self._lock:
            self._buf += chunk
            delta_chars = len(self._buf) - self._last_len
            delta_secs  = _time.monotonic() - self._last_ts
            if delta_chars >= self.MIN_CHARS or (self._last_ts > 0 and delta_secs >= self.MIN_SECS):
                await self._flush(cursor=True)

    async def _flush(self, cursor: bool) -> None:
        text = (self._prefix + self._buf + (self.CURSOR if cursor else ""))[:4096]
        try:
            await self._msg.edit_text(text, parse_mode=None)
            self._last_len = len(self._buf)
            self._last_ts  = _time.monotonic()
        except Exception:
            pass

    async def finalize(self) -> str:
        """Flush without cursor and return buffered text."""
        async with self._lock:
            await self._flush(cursor=False)
        return self._buf


# ── Per-user settings (in-memory) ─────────────────────────────────────────────
# {telegram_user_id: {"deepthink": bool}}
_settings: dict[int, dict] = {}


def _get_deepthink(uid: int) -> bool:
    return _settings.setdefault(uid, {"deepthink": True})["deepthink"]


def _set_deepthink(uid: int, val: bool) -> None:
    _settings.setdefault(uid, {})["deepthink"] = val


# ── Inline keyboard ────────────────────────────────────────────────────────────

def _keyboard(uid: int) -> InlineKeyboardMarkup:
    deepthink    = _get_deepthink(uid)
    toggle_label = "⚡ Switch to Fast" if deepthink else "🧠 Switch to Deepthink"
    mode_label   = "🧠 Deepthink ✓"   if deepthink else "⚡ Fast ✓"
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=mode_label,    callback_data="noop"),
            InlineKeyboardButton(text=toggle_label,  callback_data="toggle_mode"),
        ],
        [
            InlineKeyboardButton(text="📂 Sessions",    callback_data="session_list"),
            InlineKeyboardButton(text="➕ New Session", callback_data="new_session"),
        ],
    ])


def _sessions_keyboard(
    uid: int,
    sessions: list[dict],
    current_sid: str,
) -> InlineKeyboardMarkup:
    """Inline keyboard listing all sessions; active one gets a ✅."""
    rows = []
    for s in sessions[:15]:
        icon  = "✅" if s["id"] == current_sid else "📝"
        title = s["title"][:28]
        count = s.get("message_count", 0)
        label = f"{icon} {title} ({count} msg{'s' if count != 1 else ''})"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"switch:{s['id']}")])
    rows.append([
        InlineKeyboardButton(text="➕ New Session", callback_data="new_session"),
        InlineKeyboardButton(text="← Back",        callback_data="session_back"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


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


@router.message(Command("sessions"))
async def cmd_sessions(message: Message) -> None:
    uid      = message.from_user.id
    sessions = await get_sessions_list(str(uid))
    sid      = await get_current_session_id(str(uid))
    if not sessions:
        await message.answer(
            "No sessions yet — start chatting to create one!",
            reply_markup=_keyboard(uid),
        )
        return
    await message.answer(
        "📂 <b>Your sessions</b> — tap one to switch:",
        reply_markup=_sessions_keyboard(uid, sessions, sid),
    )


@router.message(Command("newsession"))
async def cmd_newsession(message: Message) -> None:
    uid = message.from_user.id
    await create_session(str(uid))
    await message.answer(
        "➕ <b>New session started!</b>\n"
        "Previous sessions are saved. Use /sessions to switch back.",
        reply_markup=_keyboard(uid),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "<b>Commands:</b>\n"
        "/start      — Welcome + reset mode\n"
        "/mode       — Show current mode\n"
        "/sessions   — View &amp; switch between sessions\n"
        "/newsession — Start a new session\n"
        "/help       — This message\n\n"
        "<b>Buttons on each reply:</b>\n"
        "🧠 Deepthink / ⚡ Fast  — switch reasoning mode\n"
        "📂 Sessions             — browse &amp; switch sessions\n"
        "➕ New Session          — branch into a fresh session\n\n"
        "<b>Long-term memory</b> is always active — I remember facts "
        "about you across all sessions."
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
    await create_session(str(uid))
    await cb.answer("New session started!")
    await cb.message.answer(
        "➕ <b>New session started!</b>\n"
        "Previous sessions are saved. Use /sessions to switch back.",
        reply_markup=_keyboard(uid),
    )


@router.callback_query(F.data == "session_list")
async def cb_session_list(cb: CallbackQuery) -> None:
    uid      = cb.from_user.id
    sessions = await get_sessions_list(str(uid))
    sid      = await get_current_session_id(str(uid))
    await cb.answer()
    if not sessions:
        await cb.message.answer(
            "No sessions yet — start chatting to create one!",
            reply_markup=_keyboard(uid),
        )
        return
    try:
        await cb.message.edit_text(
            "📂 <b>Your sessions</b> — tap one to switch:",
            reply_markup=_sessions_keyboard(uid, sessions, sid),
        )
    except TelegramBadRequest:
        await cb.message.answer(
            "📂 <b>Your sessions</b> — tap one to switch:",
            reply_markup=_sessions_keyboard(uid, sessions, sid),
        )


@router.callback_query(F.data.startswith("switch:"))
async def cb_switch_session(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    sid = cb.data.split(":", 1)[1]

    ok = await switch_session(str(uid), sid)
    if not ok:
        await cb.answer("Session not found!", show_alert=True)
        return

    sessions = await get_sessions_list(str(uid))
    meta  = next((s for s in sessions if s["id"] == sid), None)
    title = meta["title"] if meta else "Unknown"
    count = meta.get("message_count", 0) if meta else 0

    await cb.answer(f"Switched: {title[:30]}")
    try:
        await cb.message.edit_text(
            f"✅ Switched to session: <b>{_html.escape(title)}</b>\n"
            f"<i>{count} message{'s' if count != 1 else ''} in this session</i>",
            reply_markup=_keyboard(uid),
        )
    except TelegramBadRequest:
        await cb.message.answer(
            f"✅ Switched to session: <b>{_html.escape(title)}</b>\n"
            f"<i>{count} message{'s' if count != 1 else ''} in this session</i>",
            reply_markup=_keyboard(uid),
        )


@router.callback_query(F.data == "session_back")
async def cb_session_back(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    await cb.answer()
    try:
        await cb.message.edit_reply_markup(reply_markup=_keyboard(uid))
    except TelegramBadRequest:
        pass


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
    streamer = _TelegramStreamer(
        status,
        prefix=f"{mode_emoji} {mode_name}\n\n",
    )

    try:
        response = await process_turn(
            text, str(uid), deepthink=deepthink, stream_callback=streamer,
        )

        streamed = await streamer.finalize()

        if streamed:
            # Streaming updated the status message; attach keyboard to it
            try:
                await status.edit_reply_markup(reply_markup=_keyboard(uid))
            except TelegramBadRequest:
                pass
            # Send overflow chunks if the full response exceeds 4096 chars
            if len(response) > 4000:
                overflow = _split_4096(response[4000:])
                for i, chunk in enumerate(overflow):
                    kb = _keyboard(uid) if i == len(overflow) - 1 else None
                    try:
                        await message.answer(chunk, reply_markup=kb)
                    except TelegramBadRequest:
                        await message.answer(_html.escape(chunk), reply_markup=kb)
        else:
            # Nothing was streamed (API doesn't support it) — fall back
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
        BotCommand(command="sessions",   description="View & switch sessions"),
        BotCommand(command="newsession", description="Start a new session"),
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
