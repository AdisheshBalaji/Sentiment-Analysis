
import { useMemo } from 'react'

const MAX_BAR_HEIGHT = 80
const MIN_BAR_HEIGHT = 4

export default function AttentionHeatmap({ tokens, weights, sentiment }) {
  const isPositive = sentiment === 'Positive'

  // Normalise weights 
  const maxW = useMemo(() => Math.max(...weights, 1e-9), [weights])
  const normWeights = useMemo(() => weights.map(w => w / maxW), [weights, maxW])

  // HSL colour scale: blue for low, cyan/green/red for high depending on sentiment
  const barColor = (norm) => {
    if (isPositive) {
      // low: cold blue -> high: vivid green
      const h = 200 - norm * 80
      const s = 60 + norm * 30
      const l = 35 + norm * 20
      return `hsl(${h}, ${s}%, ${l}%)`
    } else {
      // low: cold blue → high: vivid red
      const h = 200 - norm * 200
      const s = 60 + norm * 30
      const l = 35 + norm * 20
      return `hsl(${h}, ${s}%, ${l}%)`
    }
  }

  if (!tokens || tokens.length === 0) return null

  return (
    <div className="heatmap-tokens" role="list" aria-label="Attention weights per token">
      {tokens.map((word, i) => {
        const norm = normWeights[i] ?? 0
        const barH = MIN_BAR_HEIGHT + norm * (MAX_BAR_HEIGHT - MIN_BAR_HEIGHT)
        const color = barColor(norm)
        const rawW = (weights[i] ?? 0).toFixed(4)

        return (
          <div
            key={i}
            className="token-chip"
            role="listitem"
            title={`"${word}" — attention: ${rawW}`}
            style={{ minWidth: `${Math.max(word.length * 9, 36)}px` }}
          >
            {/* Bar */}
            <div
              className="token-bar"
              style={{
                height: `${barH}px`,
                background: color,
                opacity: 0.25 + norm * 0.75,
                boxShadow: norm > 0.6 ? `0 0 10px ${color}` : 'none',
                width: '100%',
              }}
            />
            {/* Word label */}
            <span
              className="token-word"
              style={{
                background: norm > 0.6
                  ? `rgba(${isPositive ? '104,211,145' : '252,129,129'},0.08)`
                  : 'transparent',
                borderColor: norm > 0.6
                  ? `rgba(${isPositive ? '104,211,145' : '252,129,129'},0.3)`
                  : 'transparent',
                color: norm > 0.6
                  ? (isPositive ? 'var(--pos-color)' : 'var(--neg-color)')
                  : 'var(--text-primary)',
                fontWeight: norm > 0.6 ? 600 : 400,
              }}
            >
              {word}
            </span>
            {/* Weight tooltip on hover */}
            <span className="token-weight">{rawW}</span>
          </div>
        )
      })}
    </div>
  )
}
