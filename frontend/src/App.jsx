import { useCallback, useEffect, useState } from "react";
import "./App.css";

/** Empty = use Vite dev proxy (same origin). Set VITE_API_BASE=http://127.0.0.1:8000 for preview / no proxy. */
const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
const apiUrl = (path) => `${API_BASE}${path}`;
const api = (path, options) => fetch(apiUrl(path), options);

function ChunkList({ chunks }) {
  if (!chunks?.length) return <p className="muted">No chunks returned.</p>;
  return (
    <ol className="chunk-list">
      {chunks.map((c, i) => (
        <li key={i}>
          <span className="chunk-source mono">{c.source}</span>
          <p className="chunk-preview">{c.preview}</p>
        </li>
      ))}
    </ol>
  );
}

function ResultPanel({ title, colorVar, data }) {
  if (!data) return null;
  return (
    <section className="panel" style={{ "--panel-accent": `var(${colorVar})` }}>
      <h3>{title}</h3>
      <div className="answer">{data.answer || "—"}</div>
      <h4>Retrieved chunks</h4>
      <ChunkList chunks={data.chunks} />
    </section>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [query, setQuery] = useState("Impact of repo rate on EMI");
  const [bm25W, setBm25W] = useState(0.5);
  const [vecW, setVecW] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const pollHealth = useCallback(() => {
    api("/health")
      .then(async (r) => {
        const text = await r.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch {
          const hint =
            r.status >= 500
              ? " The dev server often returns this when nothing is listening on port 8000 — start the API in another terminal."
              : "";
          const preview = text.replace(/\s+/g, " ").trim().slice(0, 160);
          return {
            status: "invalid_response",
            ready: false,
            error: `Not JSON (HTTP ${r.status}).${hint}${preview ? ` Response: ${preview}${text.length > 160 ? "…" : ""}` : ""} Wrong service on this port?`,
          };
        }
        if (!r.ok) {
          const detail = data?.detail ?? data?.error;
          return {
            status: "http_error",
            ready: false,
            error: detail != null ? String(detail) : `HTTP ${r.status}: ${text.slice(0, 180)}`,
          };
        }
        return data;
      })
      .then(setHealth)
      .catch((err) =>
        setHealth({
          status: "unreachable",
          ready: false,
          error:
            "Browser could not reach the API. Fix: (1) In a separate terminal, from the project root with venv on, run: uvicorn backend.main:app --host 127.0.0.1 --port 8000  (2) Use npm run dev and open http://localhost:5173 — not file:// or preview without proxy. Optional: set VITE_API_BASE=http://127.0.0.1:8000 in frontend/.env and rebuild.",
          connectionDetail: err?.message || String(err),
        })
      );
  }, []);

  useEffect(() => {
    pollHealth();
    const t = setInterval(pollHealth, 5000);
    return () => clearInterval(t);
  }, [pollHealth]);

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await api("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          bm25_weight: bm25W,
          vector_weight: vecW,
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        const d = json.detail;
        const msg =
          typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : res.statusText;
        throw new Error(msg || "Request failed");
      }
      setResult(json);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Hybrid RAG</h1>
        <p className="tagline">BM25 + vector + ensemble — financial / banking corpus</p>
        <div className="health mono">
          {health == null && "Checking API…"}
          {health && (
            <div className="health-block">
              {health.status === "unreachable" || health.status === "invalid_response" || health.status === "http_error" ? (
                <span className="warn">
                  <strong>Cannot talk to the backend.</strong> {health.error}
                  {health.connectionDetail && (
                    <span className="health-detail"> ({health.connectionDetail})</span>
                  )}
                </span>
              ) : health.ready ? (
                <span className="ok">
                  API: ok · ready · {health.chunks_indexed ?? 0} chunks indexed
                </span>
              ) : (
                <span className="warn">
                  API: ok · <strong>not ready</strong>
                  {health.error
                    ? ` — ${health.error}`
                    : " — pipeline still loading (first start can take several minutes while embeddings are built)."}
                </span>
              )}
            </div>
          )}
        </div>
      </header>

      <form className="query-form" onSubmit={onSubmit}>
        <label htmlFor="q">Your question</label>
        <textarea
          id="q"
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. Why does EMI increase?"
        />
        <div className="weights">
          <label>
            BM25 weight
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={bm25W}
              onChange={(e) => setBm25W(Number(e.target.value))}
            />
            <span className="mono">{bm25W.toFixed(2)}</span>
          </label>
          <label>
            Vector weight
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={vecW}
              onChange={(e) => setVecW(Number(e.target.value))}
            />
            <span className="mono">{vecW.toFixed(2)}</span>
          </label>
        </div>
        <button type="submit" disabled={loading || !health?.ready}>
          {loading ? "Running…" : "Compare BM25 · Vector · Hybrid"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <h2 className="results-title mono">Query: {result.query}</h2>
          {result.weights && (
            <p className="weights-note mono">
              Hybrid weights: BM25={result.weights.bm25} · Vector={result.weights.vector}
            </p>
          )}
          <div className="grid">
            <ResultPanel title="BM25" colorVar="--bm25" data={result.bm25} />
            <ResultPanel title="Vector" colorVar="--vector" data={result.vector} />
            <ResultPanel title="Hybrid" colorVar="--hybrid" data={result.hybrid} />
          </div>

          {result.hybrid_insight && (
            <aside className="hybrid-insight">
              <h3>{result.hybrid_insight.headline}</h3>
              <ul>
                {result.hybrid_insight.bullets?.map((line, i) => (
                  <li key={i}>{line}</li>
                ))}
              </ul>
              {result.hybrid_insight.caveat && (
                <p className="insight-caveat">{result.hybrid_insight.caveat}</p>
              )}
            </aside>
          )}
        </div>
      )}

      <footer className="footer mono">
        Start API (project root): <code>uvicorn backend.main:app --reload --port 8000</code> · from{" "}
        <code>frontend/</code>: <code>npm run api</code> · both: <code>npm run dev:full</code>
      </footer>
    </div>
  );
}
