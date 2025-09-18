"use client"

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { uploadPdf, generateItemsForDocQuery } from '@/components/api'

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [owner, setOwner] = useState('demo@example.com')
  const [resp, setResp] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [showPrompt, setShowPrompt] = useState(false)
  const [topic, setTopic] = useState('')
  const [generating, setGenerating] = useState(false)
  const [genInfo, setGenInfo] = useState<any>(null)
  const router = useRouter()

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Upload PDF</h1>
      <div className="space-y-2">
        <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <input className="border p-2 w-full" placeholder="Title (optional)" value={title} onChange={(e) => setTitle(e.target.value)} />
        <input className="border p-2 w-full" placeholder="Owner email" value={owner} onChange={(e) => setOwner(e.target.value)} />
        <button
          className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
          disabled={!file || loading}
          onClick={async () => {
            if (!file) return
            setLoading(true)
            try {
              const data = await uploadPdf(file, title || undefined, owner || undefined)
              setResp(data)
            } catch (e: any) {
              setResp({ error: String(e?.response?.data?.detail || e.message) })
            } finally {
              setLoading(false)
            }
          }}
        >Upload</button>
      </div>
      {resp && (
        <div className="space-y-3">
          <div className="text-green-700 font-medium">Upload finished</div>
          <div
            className="w-40 h-40 border rounded-lg flex items-center justify-center cursor-pointer hover:bg-slate-50"
            onClick={() => setShowPrompt(true)}
            title="Click to choose what to study and start a quiz"
          >
            <div className="text-center text-sm px-2">
              <div className="font-semibold truncate">{resp?.title || 'Uploaded PDF'}</div>
              <div className="text-xs text-slate-500">doc_id</div>
              <div className="text-[10px] break-all px-1">{resp?.doc_id}</div>
            </div>
          </div>

          {showPrompt && (
            <div className="space-y-2 border rounded p-3">
              <label className="block text-sm font-medium">What do you want to study?</label>
              <input
                className="border p-2 w-full"
                placeholder="e.g., Chapter 2: Linear Algebra basics, or 'neuron activation functions'"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
              />
              <div className="flex gap-2">
                <button
                  className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
                  disabled={!topic || generating}
                  onClick={async () => {
                    if (!resp?.doc_id || !topic) return
                    setGenerating(true)
                    setGenInfo(null)
                    try {
                      const r = await generateItemsForDocQuery(resp.doc_id, topic, 1, 10)
                      setGenInfo(r)
                      // Navigate to quiz with doc_id and email prefilled
                      const email = owner || 'demo@example.com'
                      router.push(`/quiz?doc_id=${encodeURIComponent(resp.doc_id)}&email=${encodeURIComponent(email)}`)
                    } catch (e: any) {
                      setGenInfo({ error: String(e?.response?.data?.detail || e.message) })
                    } finally {
                      setGenerating(false)
                    }
                  }}
                >{generating ? 'Preparing…' : 'Generate & Start Quiz'}</button>
                <button className="px-3 py-2 border rounded" onClick={() => setShowPrompt(false)}>Cancel</button>
              </div>
              {genInfo && (
                <pre className="bg-slate-100 p-2 text-xs overflow-auto">{JSON.stringify(genInfo, null, 2)}</pre>
              )}
            </div>
          )}

          <details className="text-xs">
            <summary className="cursor-pointer select-none">Raw response</summary>
            <pre className="bg-slate-100 p-3 text-xs overflow-auto">{JSON.stringify(resp, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  )
}
