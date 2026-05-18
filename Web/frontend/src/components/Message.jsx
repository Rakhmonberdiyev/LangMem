import { useState, useEffect, useRef } from 'react'

// ── Pipeline log entry renderer ────────────────────────────────────────────────

function LogEntry({ entry }) {
  switch (entry.level) {
    case 'section':
      return (
        <div className="mt-2 first:mt-0 text-cyan-400 font-semibold border-t border-gray-700 pt-1">
          ── {entry.text}
        </div>
      )
    case 'stage':
      return (
        <div className="text-cyan-300 mt-0.5">
          {'  ▸ '}{entry.name}
          {entry.detail && <span className="text-gray-500 ml-1">{entry.detail}</span>}
        </div>
      )
    case 'kv':
      return (
        <div className="pl-4 text-gray-400">
          <span className="text-yellow-400 w-32 inline-block">{entry.label}</span>
          <span className="text-gray-300">{entry.value}</span>
        </div>
      )
    case 'ok':
      return <div className="text-green-400 mt-0.5">{'  ✓ '}{entry.text}</div>
    case 'warn':
      return <div className="text-yellow-400 mt-0.5">{'  ⚠ '}{entry.text}</div>
    case 'err':
      return <div className="text-red-400 mt-0.5">{'  ✗ '}{entry.text}</div>
    case 'timing': {
      const color = entry.ms < 500 ? 'text-green-400' : entry.ms < 2000 ? 'text-yellow-400' : 'text-red-400'
      return (
        <div className="pl-4 text-gray-500">
          <span className="w-44 inline-block">{entry.label}</span>
          <span className={color}>⏱ {entry.ms.toLocaleString()} ms</span>
        </div>
      )
    }
    case 'total_time': {
      const color = entry.ms < 3000 ? 'text-green-400' : entry.ms < 8000 ? 'text-yellow-400' : 'text-red-400'
      return (
        <div className={`mt-1 font-semibold border-t border-gray-700 pt-1 ${color}`}>
          ── ⏱ {entry.label}: {entry.ms.toLocaleString()} ms
        </div>
      )
    }
    case 'tool_call':
      return (
        <div className="mt-1">
          <div className="text-purple-400">{'  🔧 '}{entry.name}</div>
          {entry.args && (
            <div className="pl-8 text-gray-500 truncate">{entry.args}</div>
          )}
        </div>
      )
    case 'tool_result':
      return <div className="pl-8 text-gray-600 truncate">↳ {entry.preview}</div>
    case 'redis_msg':
      if (!entry.role) {
        return <div className="pl-4 text-gray-600 italic">(empty session)</div>
      }
      return (
        <div className="pl-4 flex gap-2 min-w-0">
          <span className={`flex-shrink-0 w-20 ${entry.role === 'user' ? 'text-green-500' : 'text-blue-400'}`}>
            [{entry.role}]
          </span>
          <span className="text-gray-400 truncate">{entry.preview}</span>
        </div>
      )
    default:
      return null
  }
}

// ── RAG result parser ──────────────────────────────────────────────────────────
// Each result string looks like:  "[score=0.853] filename.pdf\ntext...\n\n---\n\n..."
function parseRagHits(results) {
  const hits = []
  for (const block of results) {
    const chunks = block.split(/\n\n---\n\n/)
    for (const chunk of chunks) {
      const headerMatch = chunk.match(/^\[score=([\d.]+)\]\s*(.+?)\n([\s\S]*)$/)
      if (headerMatch) {
        hits.push({
          score: parseFloat(headerMatch[1]),
          source: headerMatch[2].trim(),
          text: headerMatch[3].trim(),
        })
      } else if (chunk.trim()) {
        hits.push({ score: null, source: 'document', text: chunk.trim() })
      }
    }
  }
  return hits
}

// ── Toggle pill button ─────────────────────────────────────────────────────────
// Full class names required (no template literals) so Tailwind JIT keeps them.
const COLOR_STYLES = {
  gray:    { pill: 'bg-gray-100 text-gray-600 hover:bg-gray-200',       badge: 'bg-gray-300 text-gray-700',          arrow: 'text-gray-400' },
  purple:  { pill: 'bg-purple-50 text-purple-700 hover:bg-purple-100',  badge: 'bg-purple-200 text-purple-800',      arrow: 'text-purple-400' },
  blue:    { pill: 'bg-blue-50 text-blue-700 hover:bg-blue-100',        badge: 'bg-blue-200 text-blue-800',          arrow: 'text-blue-400' },
  emerald: { pill: 'bg-emerald-50 text-emerald-700 hover:bg-emerald-100', badge: 'bg-emerald-200 text-emerald-800', arrow: 'text-emerald-400' },
  amber:   { pill: 'bg-amber-50 text-amber-700 hover:bg-amber-100',     badge: 'bg-amber-200 text-amber-800',        arrow: 'text-amber-400' },
  orange:  { pill: 'bg-orange-50 text-orange-700 hover:bg-orange-100',  badge: 'bg-orange-200 text-orange-800',      arrow: 'text-orange-400' },
}

function ToggleBtn({ icon, label, count, active, empty, color, onClick, liveIndicator }) {
  const styles = COLOR_STYLES[color] ?? COLOR_STYLES.gray
  const base = empty
    ? 'bg-gray-50 text-gray-400 border border-dashed border-gray-200 cursor-default'
    : styles.pill

  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full
                  text-xs font-medium transition-colors ${base}`}
    >
      <span>{icon}</span>
      <span>{label}</span>
      {!empty && count != null && (
        <span className={`${styles.badge} rounded-full px-1.5 py-0.5 text-xs font-bold`}>
          {count}
        </span>
      )}
      {liveIndicator && (
        <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
      )}
      {!empty && <span className={styles.arrow}>{active ? '▲' : '▼'}</span>}
    </button>
  )
}

// ── Metadata + pipeline log panel ─────────────────────────────────────────────
// ── Tool name formatter ────────────────────────────────────────────────────────
// "Credit_get_all_credit_name" → { server: "Credit", tool: "get_all_credit_name" }
function parseToolName(name) {
  const idx = name.indexOf('_')
  if (idx === -1) return { server: name, tool: name }
  return { server: name.slice(0, idx), tool: name.slice(idx + 1) }
}

function MetadataPanel({ metadata, log, streaming }) {
  const [showLtm,      setShowLtm]      = useState(false)
  const [showWeb,      setShowWeb]      = useState(false)
  const [showRag,      setShowRag]      = useState(false)
  const [showTools,    setShowTools]    = useState(false)
  const [showEvidence, setShowEvidence] = useState(false)
  const [showLog,      setShowLog]      = useState(false)
  const logBottomRef = useRef(null)

  const facts      = metadata?.ltm_facts   ?? []
  const webResults = metadata?.web_results ?? []
  const ragResults = metadata?.rag_results ?? []
  const toolCalls  = metadata?.tool_calls  ?? []
  const evidence   = (metadata?.evidence   ?? '').trim()
  const hasEv      = evidence && evidence !== 'No external data required.'
  const hasLog     = log && log.length > 0

  // Pipeline is done when metadata is present (non-null)
  const pipelineRan = metadata !== null

  const ragHits = parseRagHits(ragResults)

  // Auto-open log while streaming
  useEffect(() => {
    if (streaming && hasLog) setShowLog(true)
  }, [streaming, hasLog])

  // Auto-scroll log
  useEffect(() => {
    if (showLog) logBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [log, showLog])

  // Nothing to show at all (e.g. history messages without metadata)
  if (!hasLog && !pipelineRan) return null

  return (
    <div className="mt-3 pt-3 border-t border-gray-100">
      {/* Toggle pills row */}
      <div className="flex flex-wrap gap-1.5">

        {/* Pipeline log — always shown during/after streaming */}
        {hasLog && (
          <ToggleBtn
            icon="📊" label="Pipeline" count={log.length}
            active={showLog} empty={false} color="gray"
            liveIndicator={streaming}
            onClick={() => setShowLog(v => !v)}
          />
        )}

        {/* LTM — always shown once pipeline ran */}
        {pipelineRan && (
          <ToggleBtn
            icon="🧠" label="Memory" count={facts.length || null}
            active={showLtm} empty={facts.length === 0} color="purple"
            onClick={facts.length ? () => setShowLtm(v => !v) : undefined}
          />
        )}

        {/* Web search — always shown once pipeline ran */}
        {pipelineRan && (
          <ToggleBtn
            icon="🌐" label="Web Search" count={webResults.length || null}
            active={showWeb} empty={webResults.length === 0} color="blue"
            onClick={webResults.length ? () => setShowWeb(v => !v) : undefined}
          />
        )}

        {/* RAG / Document — always shown once pipeline ran */}
        {pipelineRan && (
          <ToggleBtn
            icon="📄" label="Document" count={ragHits.length || null}
            active={showRag} empty={ragResults.length === 0} color="emerald"
            onClick={ragResults.length ? () => setShowRag(v => !v) : undefined}
          />
        )}

        {/* MCP Tools — always shown once pipeline ran */}
        {pipelineRan && (
          <ToggleBtn
            icon="🔧" label="Tools" count={toolCalls.length || null}
            active={showTools} empty={toolCalls.length === 0} color="orange"
            onClick={toolCalls.length ? () => setShowTools(v => !v) : undefined}
          />
        )}

        {/* Evidence — only shown when Deepthink produced evidence */}
        {hasEv && (
          <ToggleBtn
            icon="🔍" label="Evidence" count={null}
            active={showEvidence} empty={false} color="amber"
            onClick={() => setShowEvidence(v => !v)}
          />
        )}
      </div>

      {/* Pipeline log panel */}
      {showLog && hasLog && (
        <div className="mt-2 bg-gray-900 rounded-xl p-3 max-h-72 overflow-y-auto
                        font-mono text-xs leading-5 text-gray-300">
          {log.map((entry, i) => <LogEntry key={i} entry={entry} />)}
          {streaming && (
            <span className="inline-block w-1.5 h-3 bg-green-400 animate-pulse ml-1 align-middle" />
          )}
          <div ref={logBottomRef} />
        </div>
      )}

      {/* LTM panel */}
      {showLtm && facts.length > 0 && (
        <div className="mt-2 bg-purple-50 border border-purple-100 rounded-xl p-3">
          <p className="text-xs font-semibold text-purple-700 mb-2">🧠 Long-term Memory (Mem0)</p>
          <ul className="space-y-1.5">
            {facts.map((f, i) => (
              <li key={i} className="flex gap-2 text-xs text-gray-700">
                <span className="text-purple-400 flex-shrink-0 mt-0.5">•</span>
                <span className="leading-relaxed">{f}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Web search panel */}
      {showWeb && webResults.length > 0 && (
        <div className="mt-2 bg-blue-50 border border-blue-100 rounded-xl p-3 max-h-56 overflow-y-auto">
          <p className="text-xs font-semibold text-blue-700 mb-2">🌐 Web Search Results</p>
          {webResults.map((r, i) => (
            <div key={i} className={i > 0 ? 'mt-3 pt-3 border-t border-blue-100' : ''}>
              <p className="text-xs text-gray-700 whitespace-pre-wrap leading-relaxed font-mono">
                {r}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* RAG / Document panel */}
      {showRag && ragHits.length > 0 && (
        <div className="mt-2 bg-emerald-50 border border-emerald-100 rounded-xl p-3 max-h-64 overflow-y-auto">
          <p className="text-xs font-semibold text-emerald-700 mb-2">📄 Retrieved from Documents</p>
          <div className="space-y-3">
            {ragHits.map((hit, i) => (
              <div key={i} className="bg-white rounded-lg border border-emerald-100 p-2.5">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-medium text-emerald-700 truncate max-w-[70%]">
                    {hit.source}
                  </span>
                  {hit.score != null && (
                    <span className="text-xs text-emerald-500 flex-shrink-0 ml-2">
                      {(hit.score * 100).toFixed(0)}% match
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-600 leading-relaxed whitespace-pre-wrap line-clamp-6">
                  {hit.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* MCP Tools panel */}
      {showTools && toolCalls.length > 0 && (
        <div className="mt-2 bg-orange-50 border border-orange-100 rounded-xl p-3 max-h-64 overflow-y-auto">
          <p className="text-xs font-semibold text-orange-700 mb-2">🔧 MCP Tool Results</p>
          <div className="space-y-3">
            {toolCalls.map((tc, i) => {
              const { server, tool } = parseToolName(tc.name)
              return (
                <div key={i} className="bg-white rounded-lg border border-orange-100 p-2.5">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-xs font-bold text-orange-600 bg-orange-100 rounded px-1.5 py-0.5">
                      {server}
                    </span>
                    <span className="text-xs font-medium text-gray-600 truncate">
                      {tool}
                    </span>
                  </div>
                  <p className="text-xs text-gray-600 leading-relaxed whitespace-pre-wrap line-clamp-6 font-mono">
                    {tc.result}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Deepthink Evidence panel */}
      {showEvidence && hasEv && (
        <div className="mt-2 bg-amber-50 border border-amber-100 rounded-xl p-3 max-h-56 overflow-y-auto">
          <p className="text-xs font-semibold text-amber-700 mb-2">🔍 Evidence Gathered (Deepthink)</p>
          <p className="text-xs text-gray-700 whitespace-pre-wrap leading-relaxed font-mono">
            {evidence}
          </p>
        </div>
      )}
    </div>
  )
}

// ── Message bubble ────────────────────────────────────────────────────────────
export default function Message({ message }) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-indigo-600 text-white rounded-2xl rounded-br-none px-4 py-3 shadow-sm">
          <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
            {message.content}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3 items-start">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600
                      flex items-center justify-center shadow-sm">
        <span className="text-sm">🤖</span>
      </div>

      {/* Bubble */}
      <div className="flex-1 max-w-[85%]">
        <div className="bg-white rounded-2xl rounded-bl-none px-4 py-3 shadow-sm border border-gray-100">
          {message.content ? (
            <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap break-words">
              {message.content}
              {message.streaming && (
                <span className="ml-0.5 inline-block w-0.5 h-[1.1em] bg-gray-400
                                 align-text-bottom animate-cursor-blink" />
              )}
            </p>
          ) : (
            <span className="inline-flex gap-1 items-center text-gray-400 text-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-gray-300 animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-gray-300 animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-gray-300 animate-bounce [animation-delay:300ms]" />
            </span>
          )}

          <MetadataPanel
            metadata={message.metadata}
            log={message.log}
            streaming={message.streaming}
          />
        </div>
      </div>
    </div>
  )
}
