import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchCatalogPreview,
  recordResultClick,
  runVisualSearch,
  type CatalogResponse,
  type ProductResult,
  type SearchFiltersState,
  type VisualSearchResponse,
} from '../api/visualSearch';

export const DEFAULT_FILTERS: SearchFiltersState = {
  category: '',
  shape: '',
  color: '',
  minPrice: '',
  maxPrice: '',
  inStock: true,
  exact: false,
  hybridRanking: true,
  rerank: true,
};

export function useVisualSearch() {
  const [sourceBlob, setSourceBlob] = useState<Blob | null>(null);
  const [sourceName, setSourceName] = useState<string>('query.jpg');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeBlob, setActiveBlob] = useState<Blob | null>(null);

  const [filters, setFilters] = useState<SearchFiltersState>(DEFAULT_FILTERS);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<VisualSearchResponse | null>(null);
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchCatalogPreview()
      .then(setCatalog)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const executeSearch = useCallback(
    async (blob: Blob, filename: string, currentFilters: SearchFiltersState) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setLoading(true);
      setError(null);
      try {
        const res = await runVisualSearch(blob, filename, currentFilters, controller.signal);
        setResponse(res);
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setError(err instanceof Error ? err.message : 'Search failed.');
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const selectImage = useCallback(
    (file: Blob, filename = 'upload.jpg') => {
      const nextUrl = URL.createObjectURL(file);
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return nextUrl;
      });
      setSourceBlob(file);
      setSourceName(filename);
      setActiveBlob(file);
      setResponse(null);
      setError(null);
      void executeSearch(file, filename, filters);
    },
    [executeSearch, filters],
  );

  const applyCroppedImage = useCallback(
    (croppedBlob: Blob) => {
      setActiveBlob(croppedBlob);
      void executeSearch(croppedBlob, sourceName, filters);
    },
    [executeSearch, filters, sourceName],
  );

  const resetCrop = useCallback(() => {
    if (!sourceBlob) return;
    setActiveBlob(sourceBlob);
    void executeSearch(sourceBlob, sourceName, filters);
  }, [executeSearch, filters, sourceBlob, sourceName]);

  const updateFilters = useCallback(
    (patch: Partial<SearchFiltersState>) => {
      setFilters((prev) => {
        const next = { ...prev, ...patch };
        if (activeBlob) {
          void executeSearch(activeBlob, sourceName, next);
        }
        return next;
      });
    },
    [activeBlob, executeSearch, sourceName],
  );

  const clearFilters = useCallback(() => {
    setFilters(DEFAULT_FILTERS);
    if (activeBlob) {
      void executeSearch(activeBlob, sourceName, DEFAULT_FILTERS);
    }
  }, [activeBlob, executeSearch, sourceName]);

  const clearImage = useCallback(() => {
    abortRef.current?.abort();
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setSourceBlob(null);
    setActiveBlob(null);
    setResponse(null);
    setError(null);
  }, []);

  const logClick = useCallback(
    (item: ProductResult, rank: number) => {
      if (!response) return;
      void recordResultClick({
        queryId: response.queryId,
        productId: item.productId,
        imageUrl: item.imageUrl,
        rank,
        score: item.similarity,
        modelVersion: response.modelVersion,
      });
    },
    [response],
  );

  return {
    previewUrl,
    sourceBlob,
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
    retrySearch: () => activeBlob && executeSearch(activeBlob, sourceName, filters),
  };
}
