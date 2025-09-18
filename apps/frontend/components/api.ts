import axios from 'axios'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000'

export const api = axios.create({ baseURL: API_BASE })

export async function health() {
  const { data } = await api.get('/healthz')
  return data
}

export async function search(query: string, k = 5) {
  const { data } = await api.get('/search', { params: { q: query, k } })
  return data
}

export async function uploadPdf(file: File, title?: string, owner_email?: string) {
  const form = new FormData()
  form.append('file', file)
  if (title) form.append('title', title)
  if (owner_email) form.append('owner_email', owner_email)
  const { data } = await api.post('/ingest/upload_pdf', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function quizNext(doc_id: string, user_email: string) {
  const { data } = await api.get('/quiz/next', { params: { doc_id, user_email } })
  return data
}

export async function quizAnswer(user_email: string, item_id: string, choice_index: number) {
  const { data } = await api.post('/quiz/answer', { user_email, item_id, choice_index })
  return data
}

export async function quizProgress(doc_id: string, user_email: string) {
  const { data } = await api.get('/quiz/progress', { params: { doc_id, user_email } })
  return data
}

export async function learnerState(user_email: string, doc_id?: string) {
  const { data } = await api.get('/learner/state', { params: { user_email, doc_id } })
  return data
}

export async function generateItemsForDocQuery(doc_id: string, query: string, per_chunk = 1, max_chunks = 10) {
  const { data } = await api.post('/items/generate_for_doc_query', {
    doc_id,
    query,
    per_chunk,
    max_chunks,
  })
  return data
}
