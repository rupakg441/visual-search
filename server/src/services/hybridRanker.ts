import { env } from '../config/env.js';
import type { PredictedAttributes, SearchFilters } from '../types/search.js';
import type { GroupedProduct } from './vectorSearch.js';

export interface HybridScoredProduct {
  group: GroupedProduct;
  attributeBoost: number;
}

function pickPredictedAttr(
  predicted: PredictedAttributes | null | undefined,
  key: string,
  minConfidence = 0.35,
): string | undefined {
  const entry = predicted?.[key];
  if (!entry || entry.confidence < minConfidence) return undefined;
  return String(entry.value);
}

/**
 * Switchable hybrid ranking module: combines vector similarity with attribute alignment
 * from explicit filters and/or zero-shot predicted attributes.
 */
export function computeAttributeBoosts(
  groups: GroupedProduct[],
  filters: SearchFilters,
  predictedAttributes?: PredictedAttributes | null,
  boostWeight = env.HYBRID_ATTR_BOOST,
): HybridScoredProduct[] {
  const targetShape = filters.shape ?? pickPredictedAttr(predictedAttributes, 'shape');
  const targetMetal = filters.metal ?? pickPredictedAttr(predictedAttributes, 'metal');
  const targetCategory = filters.category ?? pickPredictedAttr(predictedAttributes, 'category');

  return groups.map((group) => {
    const attrs = group.best.attributes ?? {};
    let matches = 0;
    let totalChecks = 0;

    const checks: Array<[string | undefined, unknown]> = [
      [targetShape, attrs.shape],
      [targetMetal, attrs.metal ?? attrs.color],
      [targetCategory, attrs.category],
    ];

    for (const [expected, actual] of checks) {
      if (!expected) continue;
      totalChecks += 1;
      if (typeof actual === 'string' && actual.toLowerCase() === expected.toLowerCase()) {
        matches += 1;
      }
    }

    const boost = totalChecks > 0 ? (matches / totalChecks) * boostWeight : 0;
    return {
      group,
      attributeBoost: Number(boost.toFixed(4)),
    };
  });
}
