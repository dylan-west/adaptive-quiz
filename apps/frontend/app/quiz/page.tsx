"use client"

import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { quizNext, quizAnswer, quizProgress, learnerState } from '@/components/api'

export default function QuizPage() {
  const params = useSearchParams()
  const qpDoc = params.get('doc_id') || ''
  const qpEmail = params.get('email') || ''
  const [docId, setDocId] = useState(qpDoc)
  const [email, setEmail] = useState(qpEmail || 'me@example.com')
  const [item, setItem] = useState<any>(null)
  const [status, setStatus] = useState<any>(null)

  async function getNext() {
    try {
      const data = await quizNext(docId, email)
      setItem(data)
    } catch (e: any) {
      setStatus({ error: String(e?.response?.data?.detail || e.message) })
      setItem(null)
    }
  }

  async function answer(idx: number) {
    if (!item) return
    const res = await quizAnswer(email, item.item_id, idx)
    const prog = await quizProgress(docId, email)
    const st = await learnerState(email, docId)
    setStatus({ res, prog, st })
    setItem(null)
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Quiz</h1>
      <AutoStarter docId={docId} email={email} onStart={getNext} />
      <div className="grid gap-2">
        <input className="border p-2" placeholder="doc_id" value={docId} onChange={(e) => setDocId(e.target.value)} />
        <input className="border p-2" placeholder="user_email" value={email} onChange={(e) => setEmail(e.target.value)} />
        <button className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50" disabled={!docId} onClick={getNext}>Next</button>
      </div>

      {item && (
        <div className="space-y-2 border rounded p-3">
          <div className="font-medium">{item.stem}</div>
          <div className="grid gap-2">
            {item.choices?.map((c: string, i: number) => (
              <button key={i} className="text-left border p-2 rounded hover:bg-slate-50" onClick={() => answer(i)}>
                {String.fromCharCode(65 + i)}. {c}
              </button>
            ))}
          </div>
        </div>
      )}

      {status && (
        <pre className="bg-slate-100 p-3 text-sm overflow-auto">{JSON.stringify(status, null, 2)}</pre>
      )}
    </div>
  )
}

// Auto-start if doc_id and email are present in URL
// Note: keep as a separate component to avoid useEffect conditionals inside main render
function AutoStarter({ docId, email, onStart }: { docId: string; email: string; onStart: () => void }) {
  useEffect(() => {
    if (docId && email) onStart()
  }, [docId, email])
  return null
}
