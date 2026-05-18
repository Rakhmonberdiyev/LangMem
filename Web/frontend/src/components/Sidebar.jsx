import { useState } from 'react'

function SessionItem({ session, isActive, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors group ${
        isActive
          ? 'bg-indigo-600 text-white'
          : 'text-slate-300 hover:bg-slate-700 hover:text-white'
      }`}
    >
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5 flex-shrink-0">
          {isActive ? '✅' : '💬'}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate leading-snug">
            {session.title || 'New Session'}
          </p>
          <p className={`text-xs mt-0.5 ${isActive ? 'text-indigo-200' : 'text-slate-500'}`}>
            {session.message_count ?? 0} msg{session.message_count !== 1 ? 's' : ''}
            {session.created_at ? ` · ${session.created_at}` : ''}
          </p>
        </div>
      </div>
    </button>
  )
}

export default function Sidebar({
  userId,
  onUserIdChange,
  sessions,
  currentSid,
  onSessionSwitch,
  onNewSession,
}) {
  const [draft, setDraft] = useState(userId)

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = draft.trim()
    if (trimmed) onUserIdChange(trimmed)
  }

  return (
    <aside className="w-64 flex-shrink-0 bg-slate-800 flex flex-col border-r border-slate-700">
      {/* Brand */}
      <div className="px-4 pt-5 pb-4 border-b border-slate-700">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-lg">
            🤖
          </div>
          <div>
            <h1 className="text-white font-bold text-sm leading-none">AI Dashboard</h1>
            <p className="text-slate-400 text-xs mt-0.5">Mem0 · Redis · Xazna</p>
          </div>
        </div>

        {/* User ID input */}
        <form onSubmit={handleSubmit}>
          <label className="text-xs text-slate-400 mb-1.5 block font-medium">User ID</label>
          <div className="flex gap-1.5">
            <input
              className="flex-1 min-w-0 bg-slate-700 text-white text-xs rounded-lg px-3 py-2
                         placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={draft}
              onChange={e => setDraft(e.target.value)}
              placeholder="Telegram ID or user_001"
            />
            <button
              type="submit"
              className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg px-2.5 py-2
                         text-xs font-bold transition-colors flex-shrink-0"
            >
              →
            </button>
          </div>
        </form>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-1">
        {!userId && (
          <p className="text-slate-500 text-xs text-center mt-6 px-2">
            Enter a user ID to see sessions
          </p>
        )}
        {userId && sessions.length === 0 && (
          <p className="text-slate-500 text-xs text-center mt-6 px-2">
            No sessions found
          </p>
        )}
        {sessions.map(s => (
          <SessionItem
            key={s.id}
            session={s}
            isActive={s.id === currentSid}
            onClick={() => onSessionSwitch(s.id)}
          />
        ))}
      </div>

      {/* New session button */}
      <div className="px-3 pb-4 pt-2 border-t border-slate-700">
        <button
          onClick={onNewSession}
          disabled={!userId}
          className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors
                     bg-slate-700 text-slate-200 hover:bg-slate-600
                     disabled:opacity-30 disabled:cursor-not-allowed"
        >
          + New Session
        </button>
      </div>
    </aside>
  )
}
