"use client"

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { quizNext, quizAnswer, quizProgress, quizRecap } from '@/components/api'

export default function QuizClient() {
  const params = useSearchParams()
  const qpDoc = params.get('doc_id') || ''
  const qpEmail = params.get('email') || ''
  const [docId, setDocId] = useState(qpDoc)
  const [email] = useState(qpEmail || 'demo@example.com')
  const [item, setItem] = useState<any>(null)
  const [status, setStatus] = useState<any>(null)
  const [answeredIdx, setAnsweredIdx] = useState<number | null>(null)
  const [correctIdx, setCorrectIdx] = useState<number | null>(null)
  const [recap, setRecap] = useState<any>(null)

  async function getNext() {
    try {
      const data = await quizNext(docId, email)
      setItem(data)
      setAnsweredIdx(null)
      setCorrectIdx(null)
    } catch (e: any) {
      const detail = String(e?.response?.data?.detail || e.message)
      setStatus({ error: detail })
      if (detail.includes('Quiz complete')) {
        try {
          const r = await quizRecap(docId, email)
          setRecap(r)
        } catch { /* ignore */ }
      }
      setItem(null)
    }
  }

  async function answer(idx: number) {
    if (!item) return
    const res = await quizAnswer(email, item.item_id, idx)
  // Optionally still compute progress internally (not shown to user)
  try { await quizProgress(docId, email) } catch {}
  setStatus({ res })
    setAnsweredIdx(idx)
    if (typeof res?.correct_index === 'number') setCorrectIdx(res.correct_index)
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Quiz</h1>
      <AutoStarter docId={docId} email={email} onStart={getNext} />
      <div className="grid gap-2">
        <input className="border p-2" placeholder="doc_id" value={docId} onChange={(e) => setDocId(e.target.value)} />
        <div className="flex gap-2">
          <button className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50" disabled={!docId} onClick={getNext}>Next</button>
          <button
            className="px-4 py-2 bg-slate-700 text-white rounded disabled:opacity-50"
            disabled={!docId}
            onClick={async () => {
              try {
                const r = await quizRecap(docId, email)
                setRecap(r)
              } catch (e: any) {
                setRecap({ error: String(e?.response?.data?.detail || e.message) })
              }
            }}
          >Finish</button>
        </div>
      </div>

      {item && (
        <div className="space-y-2 border rounded p-3">
          <div className="font-medium">{item.stem}</div>
          <div className="grid gap-2">
            {item.choices?.map((c: string, i: number) => (
              <button
                key={i}
                className={`text-left border p-2 rounded ${answeredIdx === null ? 'hover:bg-slate-50' : ''} ${answeredIdx !== null && i === correctIdx ? 'bg-green-50 border-green-500' : ''} ${answeredIdx !== null && i === answeredIdx && answeredIdx !== correctIdx ? 'bg-red-50 border-red-500' : ''}`}
                onClick={() => answeredIdx === null && answer(i)}
                disabled={answeredIdx !== null}
              >
                {String.fromCharCode(65 + i)}. {c}
              </button>
            ))}
          </div>
          {answeredIdx !== null && (
            <div className="text-sm mt-2">
              {answeredIdx === correctIdx ? (
                <span className="text-green-700 font-medium">Correct!</span>
              ) : (
                <span className="text-red-700 font-medium">Incorrect.</span>
              )}
              {typeof correctIdx === 'number' && (
                <span className="ml-2 text-slate-700">Correct answer: {String.fromCharCode(65 + (correctIdx ?? 0))}</span>
              )}
              {status?.res?.rationale && (
                <div className="mt-2 p-2 rounded bg-slate-50 border text-slate-700">
                  <div className="font-medium mb-1">Why</div>
                  <div>{status.res.rationale}</div>
                </div>
              )}
            </div>
          )}
          {answeredIdx !== null && (
            <div className="mt-2">
              <button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={getNext}>Next</button>
            </div>
          )}
        </div>
      )}

      {status?.error && !item && (
        <div className="p-3 border rounded bg-amber-50 text-amber-900">
          {status.error}
        </div>
      )}

      {/* User-facing feedback is shown above; JSON debug removed per 1a */}
      {recap && (
        <div className="border rounded p-3 space-y-2">
          <div className="font-semibold">Recap</div>
          {recap.error ? (
            <div className="text-red-700 text-sm">{recap.error}</div>
          ) : (
            <div className="space-y-2 text-sm">
              <div>
                <span className="font-medium">Accuracy:</span> {Math.round((recap.progress?.accuracy || 0) * 100)}% ({recap.progress?.correct}/{recap.progress?.answered} correct)
              </div>
              <div>
                <span className="font-medium">Ability (theta):</span> {recap.state?.theta} {recap.state?.theta_ci95 ? `(95% CI ${recap.state.theta_ci95[0]} to ${recap.state.theta_ci95[1]})` : ''}
              </div>
              {Array.isArray(recap.recommendations) && recap.recommendations.length > 0 && (
                <div>
                  <div className="font-medium mb-1">Study recommendations</div>
                  <ul className="list-disc pl-5 space-y-1">
                    {recap.recommendations.map((r: any, i: number) => (
                      <li key={i} className="text-slate-700">{r.snippet}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function AutoStarter({ docId, email, onStart }: { docId: string; email: string; onStart: () => void }) {
  useEffect(() => {
    if (docId && email) onStart()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docId, email])
  return null
}
