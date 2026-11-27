import { ExternalLink, Layers } from 'lucide-react';
import { resolveImageUrl, type ProductResult } from '../api/visualSearch';

interface ResultsGridProps {
  results: ProductResult[];
  loading: boolean;
  onCardClick: (item: ProductResult, rank: number) => void;
}

export function ResultsGrid({ results, loading, onCardClick }: ResultsGridProps) {
  if (loading) {
    return (
      <div className="results-grid" aria-busy="true" aria-label="Loading similar jewelry results">
        {Array.from({ length: 8 }).map((_, idx) => (
          <div key={idx} className="card">
            <div className="card-media skeleton" />
            <div className="card-body">
              <div className="skeleton" style={{ height: 16, width: '80%', borderRadius: 4 }} />
              <div className="skeleton" style={{ height: 12, width: '55%', borderRadius: 4, marginTop: 6 }} />
              <div className="skeleton" style={{ height: 18, width: '35%', borderRadius: 4, marginTop: 14 }} />
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="results-grid">
      {results.map((item, idx) => {
        const pct = (item.similarity * 100).toFixed(1);
        const imgSrc = resolveImageUrl(item);

        return (
          <article key={`${item.productId}-${item.imageUrl}`} className="card">
            <div className="card-media">
              <img
                src={imgSrc}
                alt={item.title ?? item.productId}
                loading="lazy"
                onError={(e) => {
                  if (e.currentTarget.src !== item.imageUrl) {
                    e.currentTarget.src = item.imageUrl;
                  }
                }}
              />
              <span className="similarity-badge">{pct}% match</span>
            </div>

            <div className="card-body">
              <h3 className="card-title">{item.title ?? item.productId}</h3>

              <div className="card-attrs">
                {item.attributes.category && (
                  <span className="attr-chip">{String(item.attributes.category).replace(/-/g, ' ')}</span>
                )}
                {item.attributes.shape && <span className="attr-chip">{String(item.attributes.shape)}</span>}
                {item.attributes.metal && (
                  <span className="attr-chip">{String(item.attributes.metal).replace(/-/g, ' ')}</span>
                )}
              </div>

              <div className="card-footer">
                <span className="price">
                  {item.price > 0 ? `$${item.price.toLocaleString()}` : 'View Pricing'}
                </span>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {item.matchedImagesCount > 1 && (
                    <span
                      title={`Max-pooled across ${item.matchedImagesCount} angles`}
                      style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: '0.75rem', color: '#5c5c57' }}
                    >
                      <Layers size={13} />
                      {item.matchedImagesCount}
                    </span>
                  )}

                  {item.productUrl && (
                    <a
                      href={item.productUrl}
                      target="_blank"
                      rel="noreferrer"
                      onClick={() => onCardClick(item, idx)}
                      aria-label={`View ${item.title ?? item.productId}`}
                      style={{ color: '#134023', display: 'inline-flex', alignItems: 'center', gap: 3, fontWeight: 600, textDecoration: 'none' }}
                    >
                      View <ExternalLink size={13} />
                    </a>
                  )}
                </div>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}
