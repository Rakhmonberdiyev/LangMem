import { useState, useRef, useCallback } from 'react'

export default function InputBar({ onSend, disabled, isStreaming }) {
  const [text, setText]   = useState('')
  const textareaRef       = useRef(null)

  const send = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [text, disabled, onSend])

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  function handleInput(e) {
    setText(e.target.value)
    // Auto-grow textarea up to ~160 px
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  return (
    <div className="px-4 py-4 bg-white border-t border-gray-200">
      <div className="flex gap-3 items-end max-w-4xl mx-auto">
        <textarea
          ref={textareaRef}
          value={text}
          onInput={handleInput}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={
            isStreaming
              ? 'Waiting for response…'
              : 'Message your AI — Enter to send, Shift+Enter for newline'
          }
          rows={1}
          className="flex-1 resize-none rounded-2xl border border-gray-200 px-4 py-3
                     text-sm text-gray-800 placeholder-gray-400
                     focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                     disabled:bg-gray-50 disabled:text-gray-400 transition-shadow"
        />
        <button
          onClick={send}
          disabled={!text.trim() || disabled}
          className="flex-shrink-0 w-11 h-11 rounded-2xl bg-indigo-600 hover:bg-indigo-500
                     disabled:opacity-40 disabled:cursor-not-allowed
                     text-white flex items-center justify-center text-lg
                     transition-colors shadow-sm"
          title="Send (Enter)"
        >
          {isStreaming ? (
            <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            '↑'
          )}
        </button>
      </div>
    </div>
  )
}
