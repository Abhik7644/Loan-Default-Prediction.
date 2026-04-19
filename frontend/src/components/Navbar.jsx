export default function Navbar({ page, setPage }) {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="dot" />
        LoanSense
      </div>
      <div className="navbar-links">
        <button
          className={`nav-btn ${page === "form" ? "active" : ""}`}
          onClick={() => setPage("form")}
        >
          Predict
        </button>
        <button
          className={`nav-btn ${page === "result" ? "active" : ""}`}
          onClick={() => setPage("result")}
          disabled={!page}
        >
          Result
        </button>
        <button
          className={`nav-btn ${page === "dashboard" ? "active" : ""}`}
          onClick={() => setPage("dashboard")}
        >
          Dashboard
        </button>
      </div>
    </nav>
  )
}
