function getRiskColor(risk) {
  if (!risk || risk === "N/A") return "blue"
  if (risk === "Low Risk")    return "green"
  if (risk === "Medium Risk") return "yellow"
  return "red"
}

function getVerdictClass(verdict = "") {
  if (verdict.includes("✅")) return "approve"
  if (verdict.includes("⚠"))  return "warn"
  return "reject"
}

export default function ResultPage({ result, onBack }) {
  if (!result) return (
    <div className="result-page animate-in">
      <p style={{ color: "var(--muted)" }}>No result yet. Go back and run a prediction.</p>
      <button className="back-btn" onClick={onBack}>← Back to Form</button>
    </div>
  )

  const riskColor  = getRiskColor(result.default_risk)
  const score      = result.risk_score ?? 0
  const emi        = result.emi_info
  const suggestion = result.loan_suggestion

  return (
    <div className="result-page animate-in">

      {/* Verdict banner */}
      <div className={`verdict-banner ${getVerdictClass(result.verdict)}`}>
        <div className="verdict-icon">
          {result.approved ? (result.default_risk === "High Risk" ? "❌" : result.default_risk === "Medium Risk" ? "⚠️" : "✅") : "❌"}
        </div>
        <div className="verdict-text">
          <h2>{result.default_risk !== "N/A" ? result.default_risk : "Not Eligible"}</h2>
          <p>{result.verdict?.replace(/[✅⚠️❌]/g, "").trim()}</p>
        </div>
      </div>

      {/* Metrics row */}
      <div className="result-grid">

        {/* Approval */}
        <div className="metric-card">
          <div className="metric-label">Eligibility</div>
          <div className={`metric-value ${result.approved ? "green" : "red"}`}>
            {result.approved ? "Approved" : "Rejected"}
          </div>
          <div style={{ marginTop: "0.5rem", fontSize: "0.82rem", color: "var(--muted)" }}>
            Confidence: {(result.approval_prob * 100).toFixed(1)}%
          </div>
        </div>

        {/* Risk score */}
        <div className="metric-card">
          <div className="metric-label">Risk Score</div>
          <div className={`metric-value ${riskColor}`}>
            {result.risk_score !== null ? result.risk_score : "—"}<span style={{ fontSize: "1rem", fontWeight: 400 }}>/100</span>
          </div>
          <div className="risk-bar-wrap">
            <div className="risk-bar-track">
              <div className={`risk-bar-fill ${riskColor}`} style={{ width: `${score}%` }} />
            </div>
            <div className="risk-bar-labels"><span>0</span><span>50</span><span>100</span></div>
          </div>
        </div>

        {/* Default probability */}
        <div className="metric-card">
          <div className="metric-label">Default Probability</div>
          <div className={`metric-value ${riskColor}`}>
            {result.default_prob !== null ? `${(result.default_prob * 100).toFixed(1)}%` : "—"}
          </div>
          <div style={{ marginTop: "0.5rem", fontSize: "0.82rem", color: "var(--muted)" }}>
            {result.default_risk ?? "Stage 2 skipped"}
          </div>
        </div>

        {/* Approval probability */}
        <div className="metric-card">
          <div className="metric-label">Approval Probability</div>
          <div className="metric-value blue">
            {(result.approval_prob * 100).toFixed(1)}%
          </div>
          <div style={{ marginTop: "0.5rem", fontSize: "0.82rem", color: "var(--muted)" }}>
            Based on eligibility model
          </div>
        </div>
      </div>

      {/* Reasons */}
      {result.reasons?.length > 0 && (
        <div className="card card-sm" style={{ marginBottom: "1rem" }}>
          <div className="metric-label" style={{ marginBottom: "0.25rem" }}>Issues Flagged</div>
          <ul className="reasons-list">
            {result.reasons.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </div>
      )}

      {/* EMI Info */}
      {emi && (
        <div className="card card-sm" style={{ marginBottom: "1rem" }}>
          <div className="metric-label">EMI Feasibility</div>
          <div className="emi-grid">
            <div className="emi-item">
              <div className="val">${emi.monthly_inc?.toLocaleString()}</div>
              <div className="lbl">Monthly Income</div>
            </div>
            <div className="emi-item">
              <div className="val">${emi.current_emi_burden?.toLocaleString()}</div>
              <div className="lbl">Existing Burden / mo</div>
            </div>
            <div className="emi-item">
              <div className="val">${emi.available_for_emi?.toLocaleString()}</div>
              <div className="lbl">Available for EMI</div>
            </div>
            {emi.proposed_emi && (
              <div className="emi-item">
                <div className="val" style={{ color: emi.feasible ? "var(--green)" : "var(--red)" }}>
                  ${emi.proposed_emi?.toLocaleString()}
                </div>
                <div className="lbl">Proposed EMI / mo</div>
              </div>
            )}
            <div className="emi-item">
              <div className="val">{emi.feasible ? "✅ Yes" : "❌ No"}</div>
              <div className="lbl">EMI Affordable?</div>
            </div>
            <div className="emi-item">
              <div className="val">${emi.recommended_max_loan?.toLocaleString()}</div>
              <div className="lbl">Max Loan You Can Afford</div>
            </div>
          </div>
        </div>
      )}

      {/* Loan suggestion */}
      {suggestion?.suggestion === "reduce_loan_amount" && (
        <div className="suggestion-box">
          💡 <strong>Suggestion:</strong> {suggestion.reason}
        </div>
      )}

      <button className="back-btn" onClick={onBack}>← New Prediction</button>
    </div>
  )
}
