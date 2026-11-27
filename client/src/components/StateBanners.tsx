import { AlertCircle, Search, Sparkles } from 'lucide-react';

export function ErrorBanner({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="banner error" role="alert">
      <AlertCircle size={28} style={{ marginBottom: '0.5rem' }} />
      <div style={{ fontWeight: 700, fontSize: '1.05rem' }}>Search could not be completed</div>
      <p style={{ margin: '0.4rem 0 1rem', fontSize: '0.9rem' }}>{message}</p>
      <button type="button" className="btn" onClick={onRetry}>
        Try Again
      </button>
    </div>
  );
}

export function EmptyResultsBanner({ onResetFilters }: { onResetFilters: () => void }) {
  return (
    <div className="banner" role="status">
      <Search size={28} color="#134023" style={{ marginBottom: '0.5rem' }} />
      <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#134023' }}>
        No matching jewelry found
      </div>
      <p style={{ margin: '0.4rem 0 1rem', fontSize: '0.9rem', color: '#5c5c57' }}>
        Try clearing one or more attribute or price filters to broaden the visual match.
      </p>
      <button type="button" className="btn btn-primary" onClick={onResetFilters}>
        Clear Filters
      </button>
    </div>
  );
}

export function WelcomeBanner({ catalogCount }: { catalogCount?: number }) {
  return (
    <div className="banner">
      <Sparkles size={30} color="#a88339" style={{ marginBottom: '0.55rem' }} />
      <h2 style={{ margin: 0, fontSize: '1.65rem', color: '#134023' }}>
        Find Visually Similar Diamond Jewelry
      </h2>
      <p style={{ maxWidth: 520, margin: '0.55rem auto 0', color: '#5c5c57', fontSize: '0.94rem' }}>
        Upload a studio shot, screenshot, or mobile photo of a ring, band, or setting. Our OpenCLIP
        + MongoDB vector pipeline isolates the piece, removes background clutter, and ranks closest
        matches in milliseconds.
      </p>
      {typeof catalogCount === 'number' && catalogCount > 0 && (
        <div style={{ marginTop: '0.9rem' }}>
          <span className="pill">{catalogCount}+ catalog items ready locally</span>
        </div>
      )}
    </div>
  );
}
