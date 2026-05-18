"""Rich-based terminal UI — imported by agent.py and pipeline modules."""

from contextvars import ContextVar

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.theme import Theme
from rich.table import Table

# Per-request SSE queue; set by the FastAPI handler before calling process_turn.
_log_queue: ContextVar = ContextVar('_log_queue', default=None)

# Tracks the last tool name called (for categorising tool results).
_last_tool: ContextVar = ContextVar('_last_tool', default=None)

# Per-request tool result collector: {'rag': [str, ...], 'web': [str, ...]}
_tool_collector: ContextVar = ContextVar('_tool_collector', default=None)


def set_log_queue(q) -> None:
    """Attach an asyncio.Queue to capture log events for the current async task."""
    _log_queue.set(q)


def set_tool_collector(collector: dict) -> None:
    """Attach a {'rag': [], 'web': []} dict to accumulate tool results."""
    _tool_collector.set(collector)


def _emit(**kwargs) -> None:
    q = _log_queue.get()
    if q is not None:
        try:
            q.put_nowait({"type": "log", **kwargs})
        except Exception:
            pass


_theme = Theme({
    "stage":   "bold cyan",
    "label":   "bold yellow",
    "dimval":  "dim",
    "ok":      "bold green",
    "warn":    "bold yellow",
    "err":     "bold red",
    "tool":    "bold magenta",
    "query":   "cyan",
    "reason":  "dim italic yellow",
    "user":    "bold green",
    "bot":     "bold blue",
    "section": "bold cyan",
})

console = Console(theme=_theme, highlight=False)


# ── Layout helpers ──────────────────────────────────────────────────────────────

def blank() -> None:
    console.print()


def section(title: str) -> None:
    console.rule(f"[section] {title} [/section]", style="cyan dim")
    _emit(level="section", text=title)


def user_panel(text: str) -> None:
    console.print(Panel(
        Text(text, style="white"),
        title="[user]💬  You[/user]",
        border_style="green",
        padding=(0, 2),
    ))


def response_panel(text: str) -> None:
    console.print(Panel(
        Text(text, style="white"),
        title="[bot]🤖  Assistant[/bot]",
        border_style="blue",
        padding=(0, 2),
    ))


# ── Stage / key-value ────────────────────────────────────────────────────────────

def tools_list(tools: list[dict]) -> None:
    names = [t["function"]["name"] for t in tools]
    console.print(
        f"      [label]{'Tools available':<16}[/label]  [bold]{len(names)}[/bold]  "
        f"[dim]({', '.join(names)})[/dim]"
    )


def token_usage(label: str, prompt: int, completion: int) -> None:
    """Show exact LLM token I/O for one API call (from resp.usage)."""
    total = prompt + completion
    console.print(
        f"      [label]{'🪙 Tokens':<16}[/label]  [dim]{label}[/dim]"
        f"  in=[bold cyan]{prompt:,}[/bold cyan]"
        f"  out=[bold green]{completion:,}[/bold green]"
        f"  [dim]total {total:,}[/dim]"
    )


def token_sources(sources: dict[str, int]) -> None:
    """Show estimated token breakdown by source (chars÷4 approximation)."""
    total = sum(sources.values())
    console.print(
        f"      [label]{'🪙 Context ≈ tok':<16}[/label]"
        f"  [dim]chars÷4[/dim]  total≈[bold yellow]{total:,}[/bold yellow]"
    )
    for name, count in sources.items():
        pct = count / max(total, 1) * 100
        console.print(
            f"        [dim]{name:<22}[/dim]"
            f"  [cyan]{count:>6,}[/cyan] [dim]tok  {pct:.0f}%[/dim]"
        )


def stage(name: str, detail: str = "") -> None:
    if detail:
        console.print(f"  [stage]▸ {name}[/stage]  [dimval]{detail}[/dimval]")
    else:
        console.print(f"  [stage]▸ {name}[/stage]")
    _emit(level="stage", name=name, detail=detail)


def kv(label: str, value: str, indent: int = 6) -> None:
    pad = " " * indent
    console.print(f"{pad}[label]{label:<16}[/label]  {value}")
    _emit(level="kv", label=label, value=value)


def ok(msg: str) -> None:
    console.print(f"  [ok]✓[/ok]  {msg}")
    _emit(level="ok", text=msg)


def warn(msg: str) -> None:
    console.print(f"  [warn]⚠[/warn]   {msg}")
    _emit(level="warn", text=msg)


def err(msg: str) -> None:
    console.print(f"  [err]✗[/err]  {msg}")
    _emit(level="err", text=msg)


# ── Tool calls ─────────────────────────────────────────────────────────────────

def tool_call(tool_name: str, args: str) -> None:
    console.print(
        f"  [tool]🔧 TOOL CALL → {tool_name}[/tool]\n"
        f"      [label]args:[/label] [query]{args[:300]}[/query]"
    )
    _last_tool.set(tool_name)
    _emit(level="tool_call", name=tool_name, args=args[:200])


def tool_result(text: str) -> None:
    preview = text[:1500]
    suffix = f"\n[dim]…+{len(text)-1500} chars truncated[/dim]" if len(text) > 1500 else ""
    console.print(Panel(
        Text(preview + suffix, style="dim"),
        title="[dim]↳ TOOL RESULT (sent back to LLM)[/dim]",
        border_style="dim cyan",
        padding=(0, 1),
    ))
    _emit(level="tool_result", preview=text[:200])

    collector  = _tool_collector.get()
    last_full  = _last_tool.get() or ''
    last_lower = last_full.lower()
    if collector is not None:
        if 'rag' in last_lower:
            collector['rag'].append(text)
        elif 'web' in last_lower or 'search' in last_lower:
            collector['web'].append(text)
        else:
            collector['tools'].append({"name": last_full, "result": text})


def no_tools_used() -> None:
    console.print("      [dim]  (LLM answered directly — no tools called this round)[/dim]")


# ── Timing ────────────────────────────────────────────────────────────────────

def timing(label: str, elapsed: float) -> None:
    ms = elapsed * 1000
    if ms < 500:
        color = "green"
    elif ms < 2000:
        color = "yellow"
    else:
        color = "red"
    console.print(f"      [dim]{label:<22}[/dim]  [{color}]⏱ {ms:.0f} ms[/{color}]")
    _emit(level="timing", label=label, ms=int(ms))


def total_time(label: str, elapsed: float) -> None:
    ms = elapsed * 1000
    color = "green" if ms < 3000 else ("yellow" if ms < 8000 else "red")
    console.rule(
        f"[bold {color}]⏱  {label}: {ms:.0f} ms[/bold {color}]",
        style=f"{color} dim",
    )
    _emit(level="total_time", label=label, ms=int(ms))


# ── Deepthink step I/O panels ─────────────────────────────────────────────────

def llm_input(label: str, content: str, max_chars: int = 500) -> None:
    preview = content[:max_chars].replace("\n", " ↵ ")
    if len(content) > max_chars:
        preview += f"  [dim]…+{len(content)-max_chars} chars[/dim]"
    console.print(Panel(
        Text(preview),
        title=f"[bold magenta]→ INPUT  {label}[/bold magenta]",
        border_style="magenta dim",
        padding=(0, 1),
    ))


def llm_output(label: str, content: str, max_chars: int = 600) -> None:
    preview = content[:max_chars]
    if len(content) > max_chars:
        preview += f"\n[dim]…+{len(content)-max_chars} chars[/dim]"
    console.print(Panel(
        Text(preview, style="bold white"),
        title=f"[bold cyan]← OUTPUT  {label}[/bold cyan]",
        border_style="cyan dim",
        padding=(0, 1),
    ))


def evidence_state(pieces: list[str]) -> None:
    if not pieces:
        console.print("      [dim]  Evidence: none yet[/dim]")
        return
    total = sum(len(p) for p in pieces)
    console.print(f"      [yellow]Evidence accumulated:[/yellow]  "
                  f"[bold]{len(pieces)} piece(s)[/bold]  [dim]({total} chars total)[/dim]")
    for i, p in enumerate(pieces):
        preview = p[:200].replace("\n", " ")
        if len(p) > 200:
            preview += "…"
        console.print(f"      [dim][{i+1}] {preview}[/dim]")


# ── Detail views ──────────────────────────────────────────────────────────────

def session_dump(history: list[dict]) -> None:
    """Show the Redis session turns that will be sent to the LLM."""
    if not history:
        console.print("      [dim]  (empty session)[/dim]")
        _emit(level="redis_msg", role=None, preview="(empty session)")
        return
    for msg in history:
        role = msg.get("role", "?")
        content = str(msg.get("content") or "")
        preview = content[:200].replace("\n", " ")
        if len(content) > 200:
            preview += "…"
        color = "green" if role == "user" else "blue"
        console.print(f"      [bold {color}][{role:9}][/bold {color}]  {preview}")
        _emit(level="redis_msg", role=role, preview=preview)


def ltm_dump(facts: str) -> None:
    """Show the Mem0 long-term memory facts injected into context."""
    if not facts:
        console.print("      [dim]  (no LTM facts)[/dim]")
        return
    for line in facts.splitlines():
        console.print(f"      [yellow]•[/yellow] {line.lstrip('- ')}")


def messages_dump(messages: list[dict]) -> None:
    """Show the exact messages array going into the LLM."""
    for msg in messages:
        role = msg.get("role", "?")
        content = str(msg.get("content") or "")
        preview = content[:300].replace("\n", " ↵ ")
        if len(content) > 300:
            preview += "…"
        colors = {"system": "magenta", "user": "green", "assistant": "blue", "tool": "cyan"}
        c = colors.get(role, "white")
        console.print(f"      [bold {c}][{role:9}][/bold {c}]  {preview}")


def save_redis(user_input: str, response: str) -> None:
    """Show what turn is being saved to Redis."""
    u = user_input[:150].replace("\n", " ")
    r = response[:150].replace("\n", " ")
    console.print(f"      [bold green][user     ][/bold green]  {u}")
    console.print(f"      [bold blue][assistant][/bold blue]  {r}")


def save_mem0(user_input: str, response: str) -> None:
    """Show what text is being upserted to Mem0 LTM."""
    combined = f"User: {user_input[:120]}  |  Assistant: {response[:120]}"
    console.print(f"      [yellow]{combined.replace(chr(10), ' ')}[/yellow]")


# ── Reasoning block ────────────────────────────────────────────────────────────

def reasoning_block(text: str, label: str = "Reasoning") -> None:
    if not text:
        return
    short = text[:700] + ("…" if len(text) > 700 else "")
    console.print(Panel(
        Text(short, style="dim italic"),
        title=f"[yellow]🤔 {label}[/yellow]",
        border_style="yellow dim",
        padding=(0, 1),
    ))
