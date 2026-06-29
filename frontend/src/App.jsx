import { useState, useCallback, useEffect } from 'react'
import AttentionHeatmap from './components/AttentionHeatmap'
import SentimentBadge from './components/SentimentBadge'
import ModelSelector, { MODELS, ATTENTIONS } from './components/ModelSelector'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:3000'

const SAMPLE_REVIEWS = [
  "this movie was absolutely fantastic! the acting was superb and the storyline kept me engaged throughout.",
  "terrible film. boring plot, wooden acting, and a complete waste of two hours of my life.",
  "a masterpiece of modern cinema, visually stunning with outstanding performances by the entire cast.",
]

const MODEL_LABELS = Object.fromEntries(MODELS.map(m => [m.value, m.label]))
const ATTN_LABELS = Object.fromEntries(ATTENTIONS.map(a => [a.value, a.label]))

export default function App() {
  const [text, setText] = useState('')
  const [modelName, setModelName] = useState('vanilla_lstm')
  const [attentionName, setAttnName] = useState('luong_general')
  const [available, setAvailable] = useState(new Set())
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showRaw, setShowRaw] = useState(false)

  // Fetch available combos 
  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then(r => r.json())
      .then(data => {
        const keys = (data.available_combinations ?? []).map(
          c => `${c.model_name}__${c.attention_name}`
        )
        setAvailable(new Set(keys))
      })
      .catch(() => {
      })
  }, [])

  const isAvailable = available.has(`${modelName}__${attentionName}`)

  const handleModelChange = ({ modelName: m, attentionName: a }) => {
    setModelName(m)
    setAttnName(a)
    setResult(null)
    setError(null)
  }

  const analyse = useCallback(async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    setShowRaw(false)

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_name: modelName, attention_name: attentionName }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(body.detail ?? `Server error ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to connect to the API')
    } finally {
      setLoading(false)
    }
  }, [text, modelName, attentionName])

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyse()
  }

  const useSample = (sample) => {
    setText(sample)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app-wrapper">
      <div className="container">

        {/*  Header  */}
        <header className="header">
          <div className="logo-badge">
            <span className="logo-dot" />
            {MODEL_LABELS[modelName]} · {ATTN_LABELS[attentionName]}
          </div>
          <h1>SentiScope</h1>
          <p>Select a model &amp; attention type, then analyse any movie review to see what the model attends to.</p>
        </header>

        {/*  Model Selector  */}
        <ModelSelector
          modelName={modelName}
          attentionName={attentionName}
          onChange={handleModelChange}
          available={available}
          disabled={loading}
        />

        {/*  Input Card  */}
        <div className="card input-card">
          <label htmlFor="review-input" className="input-label">Review Text</label>
          <div className="textarea-wrapper">
            <textarea
              id="review-input"
              placeholder="Type or paste a movie review… (Ctrl+Enter to analyse)"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              maxLength={1000}
            />
            <span className="char-count">{text.length} / 1000</span>
          </div>

          {/* Sample chips */}
          <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {SAMPLE_REVIEWS.map((s, i) => (
              <button
                key={i}
                onClick={() => useSample(s)}
                className="sample-chip"
              >
                Sample {i + 1}
              </button>
            ))}
          </div>

          {error && (
            <div className="error-box" role="alert">
              <span>⚠</span> {error}
            </div>
          )}

          <button
            id="analyse-btn"
            className="btn-analyse"
            onClick={analyse}
            disabled={loading || !text.trim() || !isAvailable}
          >
            <span className="btn-inner">
              {loading && <span className="spinner" aria-hidden="true" />}
              {loading
                ? 'Analysing…'
                : !isAvailable
                  ? 'Combination not yet available'
                  : 'Analyse Sentiment'}
            </span>
          </button>
        </div>

        {/*  Results  */}
        {result && (
          <div className="results-section">
            {/* Sentiment */}
            <div className="card sentiment-card">
              <div className="sentiment-info">
                <span className="section-title">Prediction</span>
                <SentimentBadge sentiment={result.sentiment} />
              </div>
              <ConfidenceRing value={result.confidence} sentiment={result.sentiment} />
            </div>

            {/* Attention Heatmap */}
            <div className="card heatmap-card">
              <div className="heatmap-header">
                <div>
                  <p className="section-title">Attention Heatmap</p>
                  <p className="heatmap-description">
                    Token height &amp; opacity reflect attention weights from{' '}
                    <strong style={{ color: 'var(--accent-cyan)' }}>
                      {ATTN_LABELS[result.attention_name]}
                    </strong>{' '}
                    on{' '}
                    <strong style={{ color: 'var(--accent-cyan)' }}>
                      {MODEL_LABELS[result.model_name]}
                    </strong>.
                    Hover a token to see its exact score.
                  </p>
                </div>
                <div className="legend">
                  <span>Low</span>
                  <div className="legend-bar" />
                  <span>High</span>
                </div>
              </div>
              <AttentionHeatmap
                tokens={result.tokens}
                weights={result.attention_weights}
                sentiment={result.sentiment}
              />
            </div>

            {/* Raw JSON */}
            <div className="raw-panel">
              <button
                className="raw-toggle"
                onClick={() => setShowRaw(v => !v)}
                id="raw-toggle-btn"
              >
                <span>{showRaw ? '▲' : '▼'}</span>
                {showRaw ? 'Hide' : 'Show'} raw API response
              </button>
              {showRaw && (
                <pre className="raw-pre">{JSON.stringify(result, null, 2)}</pre>
              )}
            </div>
          </div>
        )}

        {/*  Footer  */}
        <footer className="footer">
          <p>
            Powered by <strong>FastAPI</strong> + <strong>PyTorch</strong> ·{' '}
            <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
              API Docs ↗
            </a>
          </p>
        </footer>

      </div>
    </div>
  )
}

/*  Confidence Ring */
function ConfidenceRing({ value, sentiment }) {
  const radius = 38
  const circ = 2 * Math.PI * radius
  const pct = Math.round(value * 100)
  const dashOffset = circ * (1 - value)

  return (
    <div className="confidence-ring">
      <svg width="96" height="96" viewBox="0 0 96 96" className="ring-svg">
        <circle className="ring-track" cx="48" cy="48" r={radius} />
        <circle
          className={`ring-fill ${sentiment.toLowerCase()}`}
          cx="48"
          cy="48"
          r={radius}
          strokeDasharray={circ}
          strokeDashoffset={dashOffset}
        />
        <text
          x="48" y="44"
          textAnchor="middle"
          dominantBaseline="middle"
          fill={sentiment === 'Positive' ? '#68d391' : '#fc8181'}
          fontSize="15"
          fontWeight="700"
          fontFamily="JetBrains Mono, monospace"
          style={{ transform: 'rotate(90deg)', transformOrigin: '48px 48px' }}
        >
          {pct}%
        </text>
        <text
          x="48" y="62"
          textAnchor="middle"
          fill="#4a5568"
          fontSize="8.5"
          fontFamily="Inter, sans-serif"
          letterSpacing="0.08em"
          style={{ transform: 'rotate(90deg)', transformOrigin: '48px 48px' }}
        >
          CONF
        </text>
      </svg>
      <span className="ring-label">Confidence</span>
    </div>
  )
}
