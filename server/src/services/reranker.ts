import { env } from '../config/env.js';
import type { RankedProductResult } from '../types/search.js';
import type { HybridScoredProduct } from './hybridRanker.js';
import { cosineSimilarity } from './vectorSearch.js';

export interface RerankOptions {
  enabled: boolean;
  secondaryQueryEmbedding?: number[] | null;
  dinov2Weight?: number;
  maxCandidates?: number;
}

/**
 * Switchable top-K reranking stage combining:
 * 1) Primary vector similarity
 * 2) Optional DINOv2 secondary embedding fusion
 * 3) Multi-view product consensus (products with multiple high-scoring angles)
 * 4) Hybrid attribute alignment boost (headroom-scaled so top scores stay differentiated)
 */
export function rerankCandidates(
  scoredGroups: HybridScoredProduct[],
  options: RerankOptions,
): RankedProductResult[] {
  const limit = options.maxCandidates ?? env.RERANK_CANDIDATES;
  const subset = scoredGroups.slice(0, limit);
  const dinoWeight = options.dinov2Weight ?? env.DINOV2_WEIGHT;

  const ranked = subset.map(({ group, attributeBoost }) => {
    const best = group.best;
    let baseScore = best.vectorScore;

    if (
      options.enabled &&
      options.secondaryQueryEmbedding &&
      best.secondaryEmbedding &&
      best.secondaryEmbedding.length > 0
    ) {
      const dinoScore = cosineSimilarity(options.secondaryQueryEmbedding, best.secondaryEmbedding);
      baseScore = (1 - dinoWeight) * baseScore + dinoWeight * dinoScore;
    }

    let consensusBoost = 0;
    if (options.enabled && group.allScores.length > 1) {
      const sortedScores = [...group.allScores].sort((a, b) => b - a);
      consensusBoost = Math.max(0, sortedScores[1] - 0.5) * 0.04;
    }

    // Scale additive boosts by remaining headroom (1 - baseScore) so high matches don't collapse to 1.0
    const headroom = Math.max(0, 1 - baseScore);
    const scaledBoost = headroom * Math.min(0.85, attributeBoost * 3 + consensusBoost * 2);
    const finalScore = Math.min(1, baseScore + scaledBoost);

    return {
      productId: best.productId,
      title: best.title,
      imageUrl: best.imageUrl,
      publicImageUrl: best.publicImageUrl,
      productUrl: best.productUrl,
      matchedImagesCount: group.matchedImagesCount,
      attributes: best.attributes,
      price: best.price,
      inStock: best.inStock,
      modelVersion: best.modelVersion,
      similarity: Number(finalScore.toFixed(4)),
      vectorScore: Number(best.vectorScore.toFixed(4)),
      attributeBoost,
      rerankScore: Number(finalScore.toFixed(4)),
    };
  });

  if (options.enabled) {
    ranked.sort((a, b) => b.rerankScore - a.rerankScore);
  }
  return ranked;
}
