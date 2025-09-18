"use client"

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { uploadPdf, generateItemsForDocQuery, docsList } from '@/components/api'

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [resp, setResp] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(false)
  const [topic, setTopic] = useState('')
  const [generating, setGenerating] = useState(false)
  const [genInfo, setGenInfo] = useState<any>(null)
  const [docs, setDocs] = useState<any[]>([])
  const [ownerEmail] = useState('demo@example.com')

  useEffect(() => {
    (async () => {
      try {
        const r = await docsList(ownerEmail, 20)
        setDocs(r.docs || [])
      } catch {}
    })()
  }, [ownerEmail])
  const router = useRouter()

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Adaptive Quiz</h1>
      <div className="space-y-3">
        <div className="grid gap-2">
          <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <input className="border p-2 w-full" placeholder="Title (optional)" value={title} onChange={(e) => setTitle(e.target.value)} />
          <button
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
            disabled={!file || loading}
            onClick={async () => {
              if (!file) return
              setLoading(true)
              try {
                const data = await uploadPdf(file, title || undefined)
                setResp(data)
                setSelected(false)
                setTopic('')
              } catch (e: any) {
                setResp({ error: String(e?.response?.data?.detail || e.message) })
              } finally {
                setLoading(false)
              }
            }}
          >{loading ? 'Uploading…' : 'Upload'}</button>
        </div>

        {resp?.doc_id && (
          <div className="space-y-3">
            <div className="text-green-700 font-medium">Upload finished</div>
            <div
              className={`w-44 h-44 border rounded-lg flex items-center justify-center cursor-pointer select-none transition-colors ${selected ? 'ring-2 ring-blue-600 bg-blue-50' : 'hover:bg-slate-50'}`}
              onClick={() => setSelected((s) => !s)}
              title="Click to select this document"
            >
              <div className="text-center text-sm px-2">
                <div className="font-semibold truncate">{resp?.title || 'Uploaded PDF'}</div>
                <div className="text-xs text-slate-500">doc_id</div>
                <div className="text-[10px] break-all px-1">{resp?.doc_id}</div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium">What do you want to study?</label>
              <input
                className="border p-2 w-full"
                placeholder="e.g., Chapter 2: Linear Algebra basics, or 'neuron activation functions'"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                disabled={!selected}
              />
              <div className="flex gap-2">
                <button
                  className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
                  disabled={!selected || !topic || generating}
                  onClick={async () => {
                    try {
                      setGenerating(true)
                      setGenInfo(null)
                      const r = await generateItemsForDocQuery(resp.doc_id, topic, 1, 10)
                      setGenInfo(r)
                      router.push(`/quiz?doc_id=${encodeURIComponent(resp.doc_id)}`)
                    } catch (e: any) {
                      setGenInfo({ error: String(e?.response?.data?.detail || e.message) })
                    } finally {
                      setGenerating(false)
                    }
                  }}
                >{generating ? 'Preparing…' : 'Generate & Start Quiz'}</button>
              </div>
              {genInfo && (
                <details className="text-xs">
                  <summary className="cursor-pointer select-none">Details</summary>
                  <pre className="bg-slate-100 p-2 text-xs overflow-auto">{JSON.stringify(genInfo, null, 2)}</pre>
                </details>
              )}
            </div>
          </div>
        )}

        {/* Recent docs */}
        {docs.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium">Recent documents</div>
            <div className="flex flex-wrap gap-3">
              {docs.map((d) => (
                <div
                  key={d.doc_id}
                  className={`w-44 h-24 border rounded-lg p-2 cursor-pointer select-none transition-colors ${selected && resp?.doc_id === d.doc_id ? 'ring-2 ring-blue-600 bg-blue-50' : 'hover:bg-slate-50'}`}
                  title={d.title}
                  onClick={() => {
                    setSelected(true)
                    setResp({ doc_id: d.doc_id, title: d.title })
                  }}
                >
                  <div className="font-semibold truncate">{d.title}</div>
                  <div className="text-xs text-slate-500">items: {d.items}</div>
                  <div className="text-[10px] break-all mt-1">{d.doc_id}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
