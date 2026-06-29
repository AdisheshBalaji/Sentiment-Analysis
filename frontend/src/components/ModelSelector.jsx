

const MODELS = [
  { value: 'vanilla_lstm', label: 'Vanilla LSTM' },
  { value: 'vanilla_rnn', label: 'Vanilla RNN' },
  { value: 'bidirectional_rnn', label: 'Bidirectional RNN' },
  { value: 'bidirectional_lstm', label: 'Bidirectional LSTM' },
]

const ATTENTIONS = [
  { value: 'luong_general', label: 'Luong General' },
  { value: 'luong_concat', label: 'Luong Concat' },
  { value: 'luong_dot', label: 'Luong Dot' },
  { value: 'bahdanau', label: 'Bahdanau' },
]

export { MODELS, ATTENTIONS }

export default function ModelSelector({ modelName, attentionName, onChange, available, disabled }) {
  const key = `${modelName}__${attentionName}`
  const isAvailable = available.has(key)

  return (
    <div className="model-selector">
      <div className="selector-row">
        {/* Base Model */}
        <div className="selector-group">
          <label htmlFor="model-select" className="selector-label">Base Model</label>
          <div className="select-wrapper">
            <select
              id="model-select"
              value={modelName}
              onChange={e => onChange({ modelName: e.target.value, attentionName })}
              disabled={disabled}
              className="selector-select"
            >
              {MODELS.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <span className="select-arrow">▾</span>
          </div>
        </div>

        {/* Attention */}
        <div className="selector-group">
          <label htmlFor="attn-select" className="selector-label">Attention Type</label>
          <div className="select-wrapper">
            <select
              id="attn-select"
              value={attentionName}
              onChange={e => onChange({ modelName, attentionName: e.target.value })}
              disabled={disabled}
              className="selector-select"
            >
              {ATTENTIONS.map(a => (
                <option key={a.value} value={a.value}>{a.label}</option>
              ))}
            </select>
            <span className="select-arrow">▾</span>
          </div>
        </div>
      </div>

      {/* Availability badge */}
      <div className={`availability-badge ${isAvailable ? 'avail' : 'unavail'}`}>
        <span className="avail-dot" />
        {isAvailable
          ? 'Weights available'
          : 'No trained weights for this combination yet'}
      </div>
    </div>
  )
}
