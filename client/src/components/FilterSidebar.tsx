import type { SearchFiltersState } from '../api/visualSearch';

interface FilterSidebarProps {
  filters: SearchFiltersState;
  facets?: {
    categories: string[];
    shapes: string[];
    metals: string[];
  };
  onChange: (patch: Partial<SearchFiltersState>) => void;
  onReset: () => void;
}

const DEFAULT_CATEGORIES = ['wedding-rings', 'engagement-rings', 'necklaces', 'earrings', 'bracelets'];
const DEFAULT_SHAPES = ['round', 'oval', 'emerald', 'cushion', 'pear', 'marquise', 'princess', 'radiant'];
const DEFAULT_COLORS = ['gold', 'white', 'rose'];

export function FilterSidebar({ filters, facets, onChange, onReset }: FilterSidebarProps) {
  const categories = facets?.categories.length ? facets.categories : DEFAULT_CATEGORIES;
  const shapes = facets?.shapes.length ? facets.shapes : DEFAULT_SHAPES;

  return (
    <div className="filters-group" aria-label="Search filters">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: 700, fontSize: '0.92rem', color: '#134023' }}>Refine Results</span>
        <button
          type="button"
          onClick={onReset}
          style={{
            background: 'none',
            border: 'none',
            color: '#5c5c57',
            fontSize: '0.8rem',
            cursor: 'pointer',
            textDecoration: 'underline',
          }}
        >
          Reset
        </button>
      </div>

      <div className="field">
        <label htmlFor="filter-category">Category</label>
        <select
          id="filter-category"
          value={filters.category}
          onChange={(e) => onChange({ category: e.target.value })}
        >
          <option value="">All Categories</option>
          {categories.map((cat) => (
            <option key={cat} value={cat}>
              {cat.replace(/-/g, ' ')}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label htmlFor="filter-shape">Diamond Shape</label>
        <select
          id="filter-shape"
          value={filters.shape}
          onChange={(e) => onChange({ shape: e.target.value })}
        >
          <option value="">All Shapes</option>
          {shapes.map((shape) => (
            <option key={shape} value={shape}>
              {shape}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label htmlFor="filter-color">Metal Tone / Color</label>
        <select
          id="filter-color"
          value={filters.color}
          onChange={(e) => onChange({ color: e.target.value })}
        >
          <option value="">All Metal Tones</option>
          {DEFAULT_COLORS.map((c) => (
            <option key={c} value={c}>
              {c === 'gold' ? 'Yellow Gold' : c === 'white' ? 'White Gold / Platinum' : 'Rose Gold'}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>Price Range (USD)</label>
        <div className="price-row">
          <input
            type="number"
            min={0}
            placeholder="Min"
            aria-label="Minimum price"
            value={filters.minPrice}
            onChange={(e) => onChange({ minPrice: e.target.value })}
          />
          <input
            type="number"
            min={0}
            placeholder="Max"
            aria-label="Maximum price"
            value={filters.maxPrice}
            onChange={(e) => onChange({ maxPrice: e.target.value })}
          />
        </div>
      </div>

      <label className="toggle-row">
        <span>In Stock Only</span>
        <input
          type="checkbox"
          checked={filters.inStock}
          onChange={(e) => onChange({ inStock: e.target.checked })}
        />
      </label>

      <label className="toggle-row">
        <span>Hybrid Attribute Boost</span>
        <input
          type="checkbox"
          checked={filters.hybridRanking}
          onChange={(e) => onChange({ hybridRanking: e.target.checked })}
        />
      </label>

      <label className="toggle-row">
        <span>Top-50 Reranking</span>
        <input
          type="checkbox"
          checked={filters.rerank}
          onChange={(e) => onChange({ rerank: e.target.checked })}
        />
      </label>

      <label className="toggle-row">
        <span>Exact Search Mode</span>
        <input
          type="checkbox"
          checked={filters.exact}
          onChange={(e) => onChange({ exact: e.target.checked })}
        />
      </label>
    </div>
  );
}
