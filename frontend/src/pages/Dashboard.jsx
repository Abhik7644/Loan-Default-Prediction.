import { useEffect, useState } from "react"

const STATIC_STATS = {
  models: [
    { name: "Logistic Regression", f1: 0.4308, auc: 0.7134, accuracy: 0.6577 },
    { name: "Decision Tree",       f1: 0.3547, auc: 0.6730, accuracy: 0.7180 },
    { name: "Random Forest",       f1: 0.3385, auc: 0.6894, accuracy: 0.7635 },
  ],
  best_model: "Random Forest",
  dataset_size: 20000,
  features: 14,
  class_balance: { non_default: "80%", default: "20%" },
}

export default function Dashboard() {
  const [stats, setStats] = useState(STATIC_STATS)

  useEffect(() => {
    fetch("http://localhost:5000/api/model-stats")
      .then(r => r.json())
      .then(setStats)
      .catch(() => {/* use static fallback */})
  }, [])

  return (
    <div className="dashboard animate-in">
      <div className="page-header">
        <h1>Model Dashboard</h1>
        <p>Overview of dataset, model performance, and evaluation metrics.</p>
      </div>

      {/* Stat cards */}
      <div className="dash-grid">
        <div className="stat-card">
          <div className="stat-num">20K</div>
          <div className="stat-label">Total Records in Dataset</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">14</div>
          <div className="stat-label">Input Features</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">20%</div>
          <div className="stat-label">Default Rate (minority class)</div>
        </div>
        <div className="stat-card">
          <div className="stat-num" style={{ color: "var(--green)" }}>2</div>
          <div className="stat-label">Trained Models (Approval + Default)</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">SMOTE</div>
          <div className="stat-label">Imbalance Handling Strategy</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">0.76</div>
          <div className="stat-label">Best Model Accuracy (Random Forest)</div>
        </div>
      </div>

      {/* Model comparison table */}
      <div className="card">
        <div className="metric-label" style={{ marginBottom: "1rem" }}>Model Comparison</div>
        <table className="model-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>F1 Score</th>
              <th>ROC-AUC</th>
              <th>AUC Bar</th>
            </tr>
          </thead>
          <tbody>
            {stats.models.map(m => (
              <tr key={m.name} className={m.name === stats.best_model ? "best" : ""}>
                <td>{m.name} {m.name === stats.best_model ? "⭐" : ""}</td>
                <td>{(m.accuracy * 100).toFixed(1)}%</td>
                <td>{m.f1.toFixed(4)}</td>
                <td>{m.auc.toFixed(4)}</td>
                <td>
                  <div className="bar-wrap">
                    <div className="mini-bar-track">
                      <div className="mini-bar-fill" style={{ width: `${m.auc * 100}%` }} />
                    </div>
                    <span style={{ fontSize: "0.75rem", color: "var(--muted)", width: "36px" }}>
                      {m.auc.toFixed(2)}
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pipeline info */}
      <div className="card" style={{ marginTop: "1rem" }}>
        <div className="metric-label" style={{ marginBottom: "1rem" }}>Two-Stage Pipeline</div>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          {[
            { step: "01", title: "Eligibility Check", desc: "Rule-based filter on grade, DTI, income, revolving utilisation. Runs approval ML model." },
            { step: "02", title: "Default Risk", desc: "Random Forest estimates probability of default. Returns risk score 0–100 and Low/Medium/High label." },
            { step: "03", title: "EMI Feasibility", desc: "Checks if monthly EMI is affordable based on income and existing obligations. Suggests safer loan amount." },
          ].map(s => (
            <div key={s.step} style={{
              flex: "1 1 200px",
              background: "var(--bg3)",
              border: "1px solid var(--border)",
              borderRadius: "12px",
              padding: "1rem",
            }}>
              <div style={{ fontFamily: "Syne", fontSize: "2rem", fontWeight: 800, color: "var(--border)", lineHeight: 1, marginBottom: "0.5rem" }}>{s.step}</div>
              <div style={{ fontFamily: "Syne", fontWeight: 700, marginBottom: "0.35rem" }}>{s.title}</div>
              <div style={{ fontSize: "0.82rem", color: "var(--muted)", lineHeight: 1.5 }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
