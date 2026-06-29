
export default function SentimentBadge({ sentiment }) {
  const isPositive = sentiment === 'Positive'
  return (
    <div className={`sentiment-badge ${isPositive ? 'positive' : 'negative'}`}>
      <span className="sentiment-icon" aria-hidden="true">
        {isPositive ? 'positive' : 'negative'}
      </span>
      {sentiment}
    </div>
  )
}
