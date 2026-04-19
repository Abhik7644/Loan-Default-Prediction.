import { useState } from "react"

const DEFAULTS = {
  grade: "B",
  annual_inc: 60000,
  short_emp: 0,
  emp_length_num: 5,
  home_ownership: "RENT",
  dti: 15,
  purpose: "debt_consolidation",
  term: " 36 months",
  last_delinq_none: 1,
  revol_util: 40,
  total_rec_late_fee: 0,
  od_ratio: 0.5,
  loan_amount: 25000,
  term_months: 36,
}

export default function LoanForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(form)
  }

  return (
    <div className="form-page animate-in">
      <div className="page-header">
        <h1>Loan Eligibility Predictor</h1>
        <p>Fill in the applicant details to get an instant eligibility and default risk assessment.</p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-grid">

            {/* ── Personal & Employment ─── */}
            <div className="section-title">Personal & Employment</div>

            <div className="form-group">
              <label>Annual Income ($)</label>
              <input type="number" value={form.annual_inc}
                onChange={e => set("annual_inc", e.target.value)} min={0} required />
            </div>

            <div className="form-group">
              <label>Employment Length (years)</label>
              <input type="number" value={form.emp_length_num}
                onChange={e => set("emp_length_num", e.target.value)} min={0} max={10} required />
            </div>

            <div className="form-group">
              <label>Employed less than 1 year?</label>
              <select value={form.short_emp} onChange={e => set("short_emp", Number(e.target.value))}>
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </select>
            </div>

            <div className="form-group">
              <label>Home Ownership</label>
              <select value={form.home_ownership} onChange={e => set("home_ownership", e.target.value)}>
                <option>RENT</option>
                <option>OWN</option>
                <option>MORTGAGE</option>
              </select>
            </div>

            {/* ── Credit Profile ─── */}
            <div className="section-title">Credit Profile</div>

            <div className="form-group">
              <label>Credit Grade</label>
              <select value={form.grade} onChange={e => set("grade", e.target.value)}>
                {["A","B","C","D","E","F","G"].map(g => <option key={g}>{g}</option>)}
              </select>
            </div>

            <div className="form-group">
              <label>Debt-to-Income Ratio (%)</label>
              <input type="number" value={form.dti}
                onChange={e => set("dti", e.target.value)} min={0} max={100} step={0.1} required />
            </div>

            <div className="form-group">
              <label>Revolving Utilisation (%)</label>
              <input type="number" value={form.revol_util}
                onChange={e => set("revol_util", e.target.value)} min={0} max={100} step={0.1} required />
            </div>

            <div className="form-group">
              <label>Previous Delinquency?</label>
              <select value={form.last_delinq_none} onChange={e => set("last_delinq_none", Number(e.target.value))}>
                <option value={1}>Yes</option>
                <option value={0}>No</option>
              </select>
            </div>

            <div className="form-group">
              <label>Total Late Fees ($)</label>
              <input type="number" value={form.total_rec_late_fee}
                onChange={e => set("total_rec_late_fee", e.target.value)} min={0} step={0.01} />
            </div>

            <div className="form-group">
              <label>OD Ratio</label>
              <input type="number" value={form.od_ratio}
                onChange={e => set("od_ratio", e.target.value)} min={0} max={5} step={0.01} />
            </div>

            {/* ── Loan Details ─── */}
            <div className="section-title">Loan Details</div>

            <div className="form-group">
              <label>Loan Purpose</label>
              <select value={form.purpose} onChange={e => set("purpose", e.target.value)}>
                {["credit_card","debt_consolidation","home_improvement",
                  "major_purchase","medical","other","small_business",
                  "vacation","car","house","moving","wedding"].map(p =>
                  <option key={p} value={p}>{p.replace(/_/g," ")}</option>
                )}
              </select>
            </div>

            <div className="form-group">
              <label>Loan Term</label>
              <select value={form.term} onChange={e => {
                set("term", e.target.value)
                set("term_months", parseInt(e.target.value))
              }}>
                <option value=" 36 months">36 months</option>
                <option value=" 60 months">60 months</option>
              </select>
            </div>

            <div className="form-group full">
              <label>Requested Loan Amount ($) — optional for EMI check</label>
              <input type="number" value={form.loan_amount}
                onChange={e => set("loan_amount", e.target.value)} min={0} step={500} />
            </div>

          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? <><span className="spinner" /> Analysing...</> : "→  Run Prediction"}
          </button>
        </form>
      </div>
    </div>
  )
}
