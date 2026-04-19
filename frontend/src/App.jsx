import { useState } from "react"
import LoanForm from "./pages/LoanForm"
import ResultPage from "./pages/ResultPage"
import Dashboard from "./pages/Dashboard"
import Navbar from "./components/Navbar"
import "./index.css"

export default function App() {
  const [page, setPage] = useState("form")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handlePredict = async (formData) => {
    setLoading(true)
    try {
      const res = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      })
      const data = await res.json()
      setResult(data)
      setPage("result")
    } catch (err) {
      alert("Could not connect to backend. Make sure Flask is running on port 5000.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <Navbar page={page} setPage={setPage} />
      <main className="main-content">
        {page === "form"      && <LoanForm onSubmit={handlePredict} loading={loading} />}
        {page === "result"    && <ResultPage result={result} onBack={() => setPage("form")} />}
        {page === "dashboard" && <Dashboard />}
      </main>
    </div>
  )
}
