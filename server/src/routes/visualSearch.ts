import crypto from 'node:crypto';
import { Router } from 'express';
import type { Collection } from 'mongodb';
import { z } from 'zod';
import { env } from '../config/env.js';
import { getProductImagesCollection } from '../db/mongo.js';
import { HttpError } from '../middleware/errorHandler.js';
import { visualSearchRateLimiter } from '../middleware/rateLimit.js';
import { uploadSingleImage } from '../middleware/upload.js';
import { computeAttributeBoosts } from '../services/hybridRanker.js';
import { mlClient as defaultMlClient, type MlClient } from '../services/mlClient.js';
import { rerankCandidates } from '../services/reranker.js';
import { groupCandidatesByProduct, searchImageCandidates } from '../services/vectorSearch.js';
import type { ProductImageDoc, SearchFilters } from '../types/search.js';

const BooleanParam = z
  .union([z.boolean(), z.string()])
  .optional()
  .transform((v) => {
    if (v === undefined || v === '') return undefined;
    if (typeof v === 'boolean') return v;
    return v.toLowerCase() === 'true' || v === '1';
  });

const OptionalNumber = z
  .union([z.number(), z.string()])
  .optional()
  .transform((v) => {
    if (v === undefined || v === '') return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  });

const VisualSearchQuerySchema = z.object({
  category: z.string().trim().optional(),
  shape: z.string().trim().optional(),
  color: z.string().trim().optional(),
  metal: z.string().trim().optional(),
  minPrice: OptionalNumber,
  maxPrice: OptionalNumber,
  inStock: BooleanParam,
  modelVersion: z.string().trim().optional(),
  exact: BooleanParam,
  tta: BooleanParam,
  hybridRanking: BooleanParam,
  rerank: BooleanParam,
  limit: OptionalNumber,
});

export interface VisualSearchRouterDeps {
  getCollection?: () => Collection<ProductImageDoc>;
  mlClient?: MlClient;
  applyRateLimit?: boolean;
}

export function createVisualSearchRouter(deps: VisualSearchRouterDeps = {}): Router {
  const router = Router();
  const getCol = deps.getCollection ?? (() => getProductImagesCollection());
  const ml = deps.mlClient ?? defaultMlClient;

  const handlers = [
    ...(deps.applyRateLimit === false ? [] : [visualSearchRateLimiter]),
    uploadSingleImage.single('image'),
    async (req: any, res: any, next: any) => {
      const startedAt = performance.now();
      try {
        if (!req.file || !req.file.buffer) {
          throw new HttpError('Missing multipart image file field "image".', 400);
        }

        let rawFilters = req.body ?? {};
        if (typeof req.body?.filters === 'string') {
          try {
            rawFilters = { ...rawFilters, ...JSON.parse(req.body.filters) };
          } catch {
            throw new HttpError('Invalid JSON in "filters" field.', 400);
          }
        }

        const parsed = VisualSearchQuerySchema.parse(rawFilters);
        const filters: SearchFilters = {
          ...(parsed.category ? { category: parsed.category } : {}),
          ...(parsed.shape ? { shape: parsed.shape } : {}),
          ...(parsed.color ? { color: parsed.color } : {}),
          ...(parsed.metal ? { metal: parsed.metal } : {}),
          ...(parsed.minPrice !== undefined ? { minPrice: parsed.minPrice } : {}),
          ...(parsed.maxPrice !== undefined ? { maxPrice: parsed.maxPrice } : {}),
          ...(parsed.inStock !== undefined ? { inStock: parsed.inStock } : {}),
          ...(parsed.modelVersion ? { modelVersion: parsed.modelVersion } : {}),
        };

        const embedRes = await ml.embedImage(
          req.file.buffer,
          req.file.originalname,
          req.file.mimetype,
          { tta: parsed.tta },
        );

        const candidateLimit = env.RERANK_CANDIDATES;
        const { candidates, mode } = await searchImageCandidates({
          collection: getCol(),
          queryEmbedding: embedRes.embedding,
          // Never compare vectors from different encoder/preprocessing versions.
          filters: { ...filters, modelVersion: parsed.modelVersion ?? embedRes.modelVersion },
          exact: parsed.exact ?? false,
          numCandidates: 200,
          limit: candidateLimit,
        });

        const grouped = groupCandidatesByProduct(candidates);

        const useHybrid = parsed.hybridRanking ?? env.ENABLE_HYBRID_RANKING;
        const hybridScored = useHybrid
          ? computeAttributeBoosts(grouped, filters, embedRes.predictedAttributes)
          : grouped.map((group) => ({ group, attributeBoost: 0 }));

        const useRerank = parsed.rerank ?? env.ENABLE_RERANKING;
        const reranked = rerankCandidates(hybridScored, {
          enabled: useRerank,
          secondaryQueryEmbedding: embedRes.secondaryEmbedding,
          dinov2Weight: env.DINOV2_WEIGHT,
          maxCandidates: candidateLimit,
        });

        const topN = Math.min(Math.max(parsed.limit ?? env.TOP_RESULTS, 1), 50);
        const results = reranked.slice(0, topN);
        const latencyMs = Math.round(performance.now() - startedAt);
        const queryId = crypto.randomUUID();

        res.json({
          queryId,
          searchMode: mode,
          modelVersion: embedRes.modelVersion,
          latencyMs,
          totalCandidates: candidates.length,
          predictedAttributes: embedRes.predictedAttributes ?? null,
          results,
        });
      } catch (err) {
        next(err);
      }
    },
  ];

  router.post('/visual-search', ...(handlers as any[]));
  return router;
}
