import { Suspense } from 'react'
import QuizClient from './quiz-client'

export default function Page() {
  return (
    <Suspense fallback={<div className="p-4 text-sm text-slate-500">Loading quiz…</div>}>
      <QuizClient />
    </Suspense>
  )
}
