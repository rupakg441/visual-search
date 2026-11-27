export interface SearchFiltersState {
  category: string;
  shape: string;
  color: string;
  minPrice: string;
  maxPrice: string;
  inStock: boolean;
  exact: boolean;
  hybridRanking: boolean;
  rerank: boolean;
}

export interface ProductResult {
  productId: string;
  title?: string;
  imageUrl: string;
  publicImageUrl?: string;
  productUrl?: string;
  matchedImagesCount: number;
  attributes: {
    category?: string;
    shape?: string;
    color?: string;
    metal?: string;
    carat?: string | number;
  };
  price: number;
  inStock: boolean;
  modelVersion: string;
  similarity: number;
  vectorScore: number;
  attributeBoost: number;
  rerankScore: number;
}

export interface VisualSearchResponse {
  queryId: string;
  searchMode: 'atlas' | 'exact';
  modelVersion: string;
  latencyMs: number;
  totalCandidates: number;
  predictedAttributes: Record<string, { value: string; confidence: number }> | null;
  results: ProductResult[];
}

export interface CatalogItem {
  productId: string;
  title?: string;
  imageUrl: string;
  publicImageUrl?: string;
  attributes: Record<string, string | number | boolean>;
  price: number;
}

export interface CatalogResponse {
  count: number;
  facets: {
    categories: string[];
    shapes: string[];
    metals: string[];
    colors: string[];
  };
  items: CatalogItem[];
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';

export function resolveImageUrl(item: { publicImageUrl?: string; imageUrl: string }): string {
  // Prefer the source URL stored with the catalog record. publicImageUrl may
  // point at the API's local /images directory, which is not persisted on hosts
  // such as Render unless a persistent disk is configured.
  if (/^https?:\/\//i.test(item.imageUrl)) return item.imageUrl;
  if (item.publicImageUrl) {
    return item.publicImageUrl.startsWith('http')
      ? item.publicImageUrl
      : `${API_BASE}${item.publicImageUrl}`;
  }
  return item.imageUrl;
}

export async function fetchCatalogImage(imageUrl: string): Promise<Blob> {
  const proxyUrl = `${API_BASE}/api/catalog-image?url=${encodeURIComponent(imageUrl)}`;
  const res = await fetch(proxyUrl);
  if (!res.ok) throw new Error(`Failed to load sample image (${res.status}).`);
  return res.blob();
}

export async function runVisualSearch(
  imageBlob: Blob,
  filename: string,
  filters: SearchFiltersState,
  signal?: AbortSignal,
): Promise<VisualSearchResponse> {
  const form = new FormData();
  form.append('image', imageBlob, filename);

  if (filters.category) form.append('category', filters.category);
  if (filters.shape) form.append('shape', filters.shape);
  if (filters.color) form.append('color', filters.color);
  if (filters.minPrice) form.append('minPrice', filters.minPrice);
  if (filters.maxPrice) form.append('maxPrice', filters.maxPrice);
  if (filters.inStock) form.append('inStock', 'true');
  form.append('exact', String(filters.exact));
  form.append('hybridRanking', String(filters.hybridRanking));
  form.append('rerank', String(filters.rerank));

  const res = await fetch(`${API_BASE}/api/visual-search`, {
    method: 'POST',
    body: form,
    signal,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error ?? `Visual search failed (${res.status})`);
  }

  return (await res.json()) as VisualSearchResponse;
}

export async function fetchCatalogPreview(): Promise<CatalogResponse> {
  const res = await fetch(`${API_BASE}/api/products?limit=24`);
  if (!res.ok) {
    throw new Error(`Failed to fetch catalog (${res.status})`);
  }
  return (await res.json()) as CatalogResponse;
}

export async function recordResultClick(payload: {
  queryId: string;
  productId: string;
  imageUrl: string;
  rank: number;
  score: number;
  modelVersion: string;
}): Promise<void> {
  await fetch(`${API_BASE}/api/clicks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch(() => undefined);
}
