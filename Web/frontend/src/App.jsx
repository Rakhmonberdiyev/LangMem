import { useState, useEffect, useCallback, useRef } from 'react'
import {
  fetchSessions,
  fetchHistory,
  createSession,
  switchSession,
  streamChat,
} from './api'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'

let _msgCounter = 0
const uid = () => `msg-${++_msgCounter}-${Date.now()}`

function toLocalMessages(history) {
  return history.map(m => ({
    id: uid(),
    role: m.role,
    content: m.content,
    streaming: false,
    metadata: null,
  }))
}

export default function App() {
  const [userId,     setUserId]     = useState(() => localStorage.getItem('ai_user_id') || '')
  const [sessions,   setSessions]   = useState([])
  const [currentSid, setCurrentSid] = useState(null)
  const [messages,   setMessages]   = useState([])
  const [deepthink,  setDeepthink]  = useState(true)
  const [streaming,  setStreaming]  = useState(false)

  // Typewriter effect state — refs so setInterval can read current values without stale closure
  const _typeQueue = useRef('')   // chars waiting to be typed
  const _typeTimer = useRef(null) // active interval id

  // ── Load sessions whenever userId changes ──────────────────────────────────
  const loadSessions = useCallback(async (uid) => {
    if (!uid) { setSessions([]); return }
    try {
      const { sessions, current_session_id } = await fetchSessions(uid)
      setSessions(sessions)
      return current_session_id
    } catch (e) {
      console.error('fetchSessions failed:', e)
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('ai_user_id', userId)
    if (!userId) { setSessions([]); setCurrentSid(null); setMessages([]); return }
    loadSessions(userId).then(sid => {
      if (sid) setCurrentSid(sid)
    })
  }, [userId, loadSessions])

  // ── Load history when session changes ─────────────────────────────────────
  useEffect(() => {
    if (!currentSid || !userId) { setMessages([]); return }
    fetchHistory(userId, currentSid)
      .then(({ history }) => setMessages(toLocalMessages(history)))
      .catch(e => console.error('fetchHistory failed:', e))
  }, [currentSid, userId])

  // ── Session actions ────────────────────────────────────────────────────────
  const handleSessionSwitch = useCallback(async (sid) => {
    if (sid === currentSid) return
    await switchSession(userId, sid).catch(() => {})
    setCurrentSid(sid)
    await loadSessions(userId)
  }, [userId, currentSid, loadSessions])

  const handleNewSession = useCallback(async () => {
    if (!userId) return
    try {
      const { session_id } = await createSession(userId)
      setCurrentSid(session_id)
      setMessages([])
      await loadSessions(userId)
    } catch (e) {
      console.error('createSession failed:', e)
    }
  }, [userId, loadSessions])

  // ── Send message ───────────────────────────────────────────────────────────
  const handleSend = useCallback(async (text) => {
    if (!text || !userId || streaming) return

    const userMsgId = uid()
    const asstMsgId = uid()

    // Reset typewriter state
    if (_typeTimer.current) { clearInterval(_typeTimer.current); _typeTimer.current = null }
    _typeQueue.current = ''

    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user',      content: text, streaming: false, metadata: null, log: [] },
      { id: asstMsgId, role: 'assistant', content: '',   streaming: true,  metadata: null, log: [] },
    ])
    setStreaming(true)

    // Typewriter interval: 1 char per tick normally; 3 chars when queue > 50 to catch up
    _typeTimer.current = setInterval(() => {
      if (_typeQueue.current.length === 0) return
      const batch = _typeQueue.current.length > 50 ? 3 : 1
      const chars = _typeQueue.current.slice(0, batch)
      _typeQueue.current = _typeQueue.current.slice(batch)
      setMessages(prev =>
        prev.map(m => m.id === asstMsgId ? { ...m, content: m.content + chars } : m),
      )
    }, 12)

    await streamChat(userId, currentSid, text, deepthink, {
      onToken: (chunk) => {
        _typeQueue.current += chunk
      },
      onDone: (response, metadata) => {
        // Stop typing animation and show the complete final response immediately
        clearInterval(_typeTimer.current)
        _typeTimer.current = null
        _typeQueue.current = ''
        setMessages(prev =>
          prev.map(m =>
            m.id === asstMsgId
              ? { ...m, content: response, streaming: false, metadata: metadata ?? null }
              : m,
          ),
        )
        setStreaming(false)
        loadSessions(userId)
      },
      onError: (err) => {
        clearInterval(_typeTimer.current)
        _typeTimer.current = null
        _typeQueue.current = ''
        setMessages(prev =>
          prev.map(m =>
            m.id === asstMsgId
              ? { ...m, content: `⚠️ Error: ${err}`, streaming: false }
              : m,
          ),
        )
        setStreaming(false)
      },
      onLog: (entry) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === asstMsgId
              ? { ...m, log: [...(m.log ?? []), entry] }
              : m,
          ),
        )
      },
    })
  }, [userId, currentSid, deepthink, streaming, loadSessions])

  const currentSession = sessions.find(s => s.id === currentSid) ?? null

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        userId={userId}
        onUserIdChange={setUserId}
        sessions={sessions}
        currentSid={currentSid}
        onSessionSwitch={handleSessionSwitch}
        onNewSession={handleNewSession}
      />
      <ChatPanel
        messages={messages}
        isStreaming={streaming}
        deepthink={deepthink}
        onDeepthinkChange={setDeepthink}
        onSend={handleSend}
        currentSession={currentSession}
        hasUserId={!!userId}
      />
    </div>
  )
}
