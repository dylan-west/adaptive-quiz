"use client"

import { useState } from 'react'
import { search } from '@/components/api'

export default function SearchPage() {
  const [q, setQ] = useState('')
  const [res, setRes] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Search</h1>
      <div className="flex gap-2">
        <input className="border p-2 w-full" placeholder="Query..." value={q} onChange={(e) => setQ(e.target.value)} />
        <button
          className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
          disabled={!q || loading}
          onClick={async () => {
            setLoading(true)
            try {
              const data = await search(q, 5)
              setRes(data)
            } catch (e: any) {
              setRes({ error: String(e?.response?.data?.detail || e.message) })
            } finally {
              setLoading(false)
            }
          }}
        >Search</button>
      </div>
      {res && (
        <pre className="bg-slate-100 p-3 text-sm overflow-auto">{JSON.stringify(res, null, 2)}</pre>
      )}
    </div>
  )
}
