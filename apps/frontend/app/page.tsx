export default function Home() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Adaptive Quiz</h1>
      <p>Upload a PDF, search, and take an adaptive quiz.</p>
      <ul className="list-disc ml-6">
        <li>Upload: extract text (OCR if needed), embed, and store</li>
        <li>Search: vector or text search</li>
        <li>Quiz: adaptive item selection and ability updates</li>
      </ul>
    </div>
  )
}
