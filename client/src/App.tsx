import { FilterSidebar } from './components/FilterSidebar';
import { ImageCropper } from './components/ImageCropper';
import { ImageUploader } from './components/ImageUploader';
import { ResultsGrid } from './components/ResultsGrid';
import { EmptyResultsBanner, ErrorBanner, WelcomeBanner } from './components/StateBanners';
import { useVisualSearch } from './hooks/useVisualSearch';

export function App() {
  const {
    previewUrl,
    filters,
    loading,
    error,
    response,
    catalog,
    selectImage,
    applyCroppedImage,
    resetCrop,
    updateFilters,
    clearFilters,
    clearImage,
    logClick,
    retrySearch,
  } = useVisualSearch();

  return (
    <div className="app-shell">
      <div className="topbar">
        Diamond Forest Visual Search Studio · Powered by OpenCLIP + MongoDB Vector Search
      </div>

      <header className="header">
        <div className="brand">
          <h1>Diamond Forest</h1>
          <span>Visual Match</span>
        </div>
        {response && (
          <div className="meta-pills" aria-live="polite">
            <span className="pill">{response.results.length} matches</span>
            <span className="pill">{response.latencyMs} ms</span>
            <span className="pill">mode: {response.searchMode}</span>
            <span className="pill">{response.modelVersion}</span>
          </div>
        )}
      </header>

      <main className="main-grid">
        <aside className="panel" aria-label="Image query and filters">
          {!previewUrl ? (
            <ImageUploader
              onSelectFile={selectImage}
              sampleItems={catalog?.items ?? []}
              disabled={loading}
            />
          ) : (
            <ImageCropper
              imageUrl={previewUrl}
              onApplyCrop={applyCroppedImage}
              onResetCrop={resetCrop}
              onClearImage={clearImage}
              disabled={loading}
            />
          )}

          <FilterSidebar
            filters={filters}
            facets={catalog?.facets}
            onChange={updateFilters}
            onReset={clearFilters}
          />
        </aside>

        <section aria-label="Visual search results">
          {response?.predictedAttributes && (
            <div className="results-header">
              <div style={{ fontWeight: 600, fontSize: '0.92rem', color: '#5c5c57' }}>
                Detected Visual Cues:
              </div>
              <div className="meta-pills">
                {Object.entries(response.predictedAttributes).map(([key, pred]) => (
                  <span key={key} className="pill">
                    {key}: {pred.value} ({Math.round(pred.confidence * 100)}%)
                  </span>
                ))}
              </div>
            </div>
          )}

          {error ? (
            <ErrorBanner message={error} onRetry={retrySearch} />
          ) : loading ? (
            <ResultsGrid results={[]} loading={true} onCardClick={logClick} />
          ) : !response ? (
            <WelcomeBanner catalogCount={catalog?.count} />
          ) : response.results.length === 0 ? (
            <EmptyResultsBanner onResetFilters={clearFilters} />
          ) : (
            <ResultsGrid results={response.results} loading={false} onCardClick={logClick} />
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
