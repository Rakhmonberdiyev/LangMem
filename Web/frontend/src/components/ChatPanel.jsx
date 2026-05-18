import { useEffect, useRef } from 'react'
import Message from './Message'
import InputBar from './InputBar'

function EmptyState({ hasUserId }) {
  if (!hasUserId) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-8">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600
                        flex items-center justify-center text-3xl shadow-lg mb-4">
          🤖
        </div>
        <h2 className="text-xl font-semibold text-gray-700 mb-2">AI Agent Dashboard</h2>
        <p className="text-gray-400 text-sm max-w-xs">
          Enter your User ID in the sidebar to view your sessions and chat history.
          Use your Telegram user ID to see messages from your Telegram bot.
        </p>
      </div>
    )
  }
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8">
      <div className="text-5xl mb-4">💬</div>
      <h2 className="text-lg font-semibold text-gray-700 mb-2">Start a conversation</h2>
      <p className="text-gray-400 text-sm max-w-xs">
        Messages sync with your Telegram bot via the same Redis and Mem0 instances.
      </p>
    </div>
  )
}

export default function ChatPanel({
  messages,
  isStreaming,
  deepthink,
  onDeepthinkChange,
  onSend,
  currentSession,
  hasUserId,
}) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex-1 flex flex-col bg-gray-50 min-w-0">
      {/* Header */}
      <header className="flex-shrink-0 px-6 py-4 bg-white border-b border-gray-200 flex items-center justify-between">
        <div className="min-w-0">
          <h2 className="font-semibold text-gray-800 truncate">
            {currentSession ? currentSession.title : 'No session selected'}
          </h2>
          {currentSession && (
            <p className="text-xs text-gray-400 mt-0.5">
              {currentSession.message_count ?? 0} messages · {currentSession.created_at}
            </p>
          )}
        </div>

        {/* Mode toggle */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-gray-400 hidden sm:inline">Mode</span>
          <button
            onClick={() => onDeepthinkChange(!deepthink)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold
                        transition-colors ${
              deepthink
                ? 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
                : 'bg-amber-100 text-amber-700 hover:bg-amber-200'
            }`}
          >
            {deepthink ? '🧠 Deepthink' : '⚡ Fast'}
          </button>
        </div>
      </header>

      {/* Message area */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-5">
          {messages.length === 0 ? (
            <EmptyState hasUserId={hasUserId} />
          ) : (
            messages.map(msg => <Message key={msg.id} message={msg} />)
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <InputBar
        onSend={onSend}
        disabled={!hasUserId || isStreaming}
        isStreaming={isStreaming}
      />
    </div>
  )
}
