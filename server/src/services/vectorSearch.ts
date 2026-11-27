import type { Collection, Document } from 'mongodb';
import { env } from '../config/env.js';
import type { ImageCandidate, ProductImageDoc, SearchFilters } from '../types/search.js';

export function cosineSimilarity(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i += 1) {
    const va = a[i];
    const vb = b[i];
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom < 1e-12) return 0;
  // Clamp to [-1, 1] and map cosine similarity to [0, 1] friendly range if needed;
  // Atlas cosine similarity returns (1 + cosine) / 2 in [0, 1].
  const rawCos = Math.max(-1, Math.min(1, dot / denom));
  return (1 + rawCos) / 2;
}

export function buildMongoFilter(filters: SearchFilters): Document {
  const query: Document = {};
  if (filters.category) query['attributes.category'] = filters.category;
  if (filters.shape) query['attributes.shape'] = filters.shape;
  if (filters.color) query['attributes.color'] = filters.color;
  if (filters.metal) query['attributes.metal'] = filters.metal;
  if (filters.inStock !== undefined) query.inStock = filters.inStock;
  if (filters.modelVersion) query.modelVersion = filters.modelVersion;

  if (filters.minPrice !== undefined || filters.maxPrice !== undefined) {
    const priceRange: Document = {};
    if (filters.minPrice !== undefined) priceRange.$gte = filters.minPrice;
    if (filters.maxPrice !== undefined) priceRange.$lte = filters.maxPrice;
    query.price = priceRange;
  }
  return query;
}

export interface SearchCandidatesOptions {
  collection: Collection<ProductImageDoc>;
  queryEmbedding: number[];
  filters: SearchFilters;
  exact?: boolean;
  numCandidates?: number;
  limit?: number;
}

export async function searchImageCandidates({
  collection,
  queryEmbedding,
  filters,
  exact = false,
  numCandidates = 200,
  limit = 50,
}: SearchCandidatesOptions): Promise<{ candidates: ImageCandidate[]; mode: 'atlas' | 'exact' }> {
  const mongoFilter = buildMongoFilter(filters);
  const useExact = exact || env.VECTOR_SEARCH_MODE === 'exact';

  if (useExact) {
    const docs = await collection
      .find(mongoFilter, {
        projection: {
          productId: 1,
          title: 1,
          imageUrl: 1,
          localPath: 1,
          publicImageUrl: 1,
          productUrl: 1,
          attributes: 1,
          price: 1,
          inStock: 1,
          modelVersion: 1,
          embedding: 1,
          secondaryEmbedding: 1,
        },
      })
      .toArray();

    const scored: ImageCandidate[] = docs
      .filter((doc) => Array.isArray(doc.embedding) && doc.embedding.length > 0)
      .map((doc) => ({
        productId: doc.productId,
        title: doc.title,
        imageUrl: doc.imageUrl,
        localPath: doc.localPath,
        publicImageUrl: doc.publicImageUrl,
        productUrl: doc.productUrl,
        attributes: doc.attributes ?? {},
        price: doc.price ?? 0,
        inStock: doc.inStock ?? true,
        modelVersion: doc.modelVersion,
        vectorScore: cosineSimilarity(queryEmbedding, doc.embedding),
        secondaryEmbedding: doc.secondaryEmbedding,
      }))
      .sort((a, b) => b.vectorScore - a.vectorScore)
      .slice(0, limit);

    return { candidates: scored, mode: 'exact' };
  }

  const vectorSearchStage: Document = {
    $vectorSearch: {
      index: env.ATLAS_VECTOR_INDEX,
      path: 'embedding',
      queryVector: queryEmbedding,
      numCandidates,
      limit,
      ...(Object.keys(mongoFilter).length > 0 ? { filter: mongoFilter } : {}),
    },
  };

  const pipeline: Document[] = [
    vectorSearchStage,
    {
      $project: {
        _id: 0,
        productId: 1,
        title: 1,
        imageUrl: 1,
        localPath: 1,
        publicImageUrl: 1,
        productUrl: 1,
        attributes: 1,
        price: 1,
        inStock: 1,
        modelVersion: 1,
        secondaryEmbedding: 1,
        vectorScore: { $meta: 'vectorSearchScore' },
      },
    },
  ];

  const candidates = (await collection.aggregate<ImageCandidate>(pipeline).toArray()).map((c) => ({
    ...c,
    attributes: c.attributes ?? {},
  }));
  return { candidates, mode: 'atlas' };
}

export interface GroupedProduct {
  best: ImageCandidate;
  allScores: number[];
  matchedImagesCount: number;
}

export function groupCandidatesByProduct(candidates: ImageCandidate[]): GroupedProduct[] {
  const byProduct = new Map<string, GroupedProduct>();

  for (const cand of candidates) {
    const existing = byProduct.get(cand.productId);
    if (!existing) {
      byProduct.set(cand.productId, {
        best: cand,
        allScores: [cand.vectorScore],
        matchedImagesCount: 1,
      });
    } else {
      existing.allScores.push(cand.vectorScore);
      existing.matchedImagesCount += 1;
      if (cand.vectorScore > existing.best.vectorScore) {
        existing.best = cand;
      }
    }
  }

  return Array.from(byProduct.values()).sort((a, b) => b.best.vectorScore - a.best.vectorScore);
}
