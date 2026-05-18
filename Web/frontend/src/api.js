const BASE = '/api'

async function _json(url, opts = {}) {
  const r = await fetch(url, opts)
  if (!r.ok) {
    const text = await r.text()
    throw new Error(`${r.status}: ${text}`)
  }
  return r.json()
}

export function fetchSessions(userId) {
  return _json(`${BASE}/sessions?user_id=${encodeURIComponent(userId)}`)
}

export function fetchHistory(userId, sessionId) {
  return _json(`${BASE}/sessions/${sessionId}/history?user_id=${encodeURIComponent(userId)}`)
}

export function createSession(userId) {
  return _json(`${BASE}/sessions?user_id=${encodeURIComponent(userId)}`, { method: 'POST' })
}

export function switchSession(userId, sessionId) {
  return _json(
    `${BASE}/sessions/${sessionId}/switch?user_id=${encodeURIComponent(userId)}`,
    { method: 'POST' },
  )
}

/**
 * Stream a chat turn via Server-Sent Events over POST.
 *
 * callbacks:
 *   onToken(chunk)          — called for each text token
 *   onDone(response, meta)  — called when pipeline finishes
 *   onError(message)        — called on error
 */
export async function streamChat(userId, sessionId, message, deepthink, { onToken, onDone, onError, onLog }) {
  try {
    const response = await fetch(`${BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, session_id: sessionId, message, deepthink }),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      // SSE events are separated by double newlines
      const parts = buffer.split('\n\n')
      buffer = parts.pop() // last item may be incomplete

      for (const part of parts) {
        const dataLine = part.split('\n').find(l => l.startsWith('data: '))
        if (!dataLine) continue
        try {
          const data = JSON.parse(dataLine.slice(6))
          if (data.type === 'token') onToken(data.content)
          else if (data.type === 'done') onDone(data.response, data.metadata)
          else if (data.type === 'error') onError(data.error)
          else if (data.type === 'log' && onLog) onLog(data)
        } catch {
          // ignore malformed events
        }
      }
    }
  } catch (err) {
    onError(err.message)
  }
}
