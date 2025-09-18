import './globals.css'
import type { Metadata } from 'next'
import Link from 'next/link'

export const metadata: Metadata = {
  title: 'Adaptive Quiz',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900">
        <nav className="border-b bg-white">
          <div className="mx-auto max-w-4xl px-4 py-3 flex gap-4">
            <Link href="/">Home</Link>
            <Link href="/upload">Upload</Link>
            <Link href="/search">Search</Link>
            <Link href="/quiz">Quiz</Link>
          </div>
        </nav>
        <main className="mx-auto max-w-4xl p-4">{children}</main>
      </body>
    </html>
  )
}
