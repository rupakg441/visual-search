import { describe, expect, it, vi } from 'vitest';
import request from 'supertest';
import { createApp } from '../src/app.js';
import type { MlClient } from '../src/services/mlClient.js';
import type { ProductImageDoc } from '../src/types/search.js';

function createMockCollection(docs: ProductImageDoc[]) {
  return {
    find: vi.fn((filter: Record<string, unknown>) => {
      const filtered = docs.filter((d) => {
        if (filter['attributes.category'] && d.attributes.category !== filter['attributes.category']) {
          return false;
        }
        if (filter['attributes.shape'] && d.attributes.shape !== filter['attributes.shape']) {
          return false;
        }
        if (filter.inStock !== undefined && d.inStock !== filter.inStock) {
          return false;
        }
        const priceFilter = filter.price as { $gte?: number; $lte?: number } | undefined;
        if (priceFilter?.$gte !== undefined && d.price < priceFilter.$gte) return false;
        if (priceFilter?.$lte !== undefined && d.price > priceFilter.$lte) return false;
        return true;
      });
      return {
        toArray: async () => filtered,
      };
    }),
    aggregate: vi.fn(() => ({
      toArray: async () => [],
    })),
  } as any;
}

describe('POST /api/visual-search', () => {
  const sampleDocs: ProductImageDoc[] = [
    {
      productId: 'df-lr3529',
      title: 'Carmen Marquise Engagement Ring',
      imageUrl: 'https://cdn.diamondforest.com/Products/Rings/LR3529/LR3529_DMQ_20/LR3529_DMQ_20_GOLD_1.webp',
      publicImageUrl: '/images/df-lr3529/LR3529_DMQ_20_GOLD_1.webp',
      embedding: [1, 0, 0, 0],
      attributes: { category: 'engagement-rings', shape: 'marquise', metal: 'yellow-gold', color: 'gold' },
      price: 2450,
      inStock: true,
      modelVersion: 'test-v1',
      createdAt: new Date(),
    },
    {
      productId: 'df-lr3529',
      title: 'Carmen Marquise Engagement Ring',
      imageUrl: 'https://cdn.diamondforest.com/Products/Rings/LR3529/LR3529_DMQ_20/LR3529_DMQ_20_GOLD_2.webp',
      publicImageUrl: '/images/df-lr3529/LR3529_DMQ_20_GOLD_2.webp',
      embedding: [0.95, 0.05, 0, 0],
      attributes: { category: 'engagement-rings', shape: 'marquise', metal: 'yellow-gold', color: 'gold' },
      price: 2450,
      inStock: true,
      modelVersion: 'test-v1',
      createdAt: new Date(),
    },
    {
      productId: 'df-lr5107',
      title: 'Penstemon Round Ring',
      imageUrl: 'https://cdn.diamondforest.com/Products/Rings/LR5107/LR5107_BR_3/LR5107_BR_3_WHITE_1.webp',
      embedding: [0, 1, 0, 0],
      attributes: { category: 'engagement-rings', shape: 'round', metal: 'white-gold', color: 'white' },
      price: 1850,
      inStock: true,
      modelVersion: 'test-v1',
      createdAt: new Date(),
    },
  ];

  const mockMl: MlClient = {
    embedImage: vi.fn(async () => ({
      embedding: [1, 0, 0, 0],
      dims: 4,
      modelVersion: 'test-v1',
      predictedAttributes: {
        shape: { value: 'marquise', confidence: 0.91 },
        metal: { value: 'yellow-gold', confidence: 0.85 },
      },
    })),
    health: vi.fn(async () => ({ status: 'ok' })),
  };

  it('groups multi-vector product images, applies filters, and returns top matches', async () => {
    const app = createApp({
      getCollection: () => createMockCollection(sampleDocs),
      mlClient: mockMl,
      applyRateLimit: false,
      checkMongoHealth: async () => true,
    });

    const res = await request(app)
      .post('/api/visual-search')
      .field('exact', 'true')
      .field('category', 'engagement-rings')
      .attach('image', Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
        filename: 'query.png',
        contentType: 'image/png',
      });

    expect(res.status).toBe(200);
    expect(res.body.searchMode).toBe('exact');
    expect(res.body.results).toHaveLength(2);
    expect(res.body.results[0].productId).toBe('df-lr3529');
    expect(res.body.results[0].matchedImagesCount).toBe(2);
    expect(res.body.results[0].similarity).toBeGreaterThan(res.body.results[1].similarity);
  });

  it('rejects missing image file or unsupported mime type with 400', async () => {
    const app = createApp({
      getCollection: () => createMockCollection(sampleDocs),
      mlClient: mockMl,
      applyRateLimit: false,
      checkMongoHealth: async () => true,
    });

    const missing = await request(app).post('/api/visual-search');
    expect(missing.status).toBe(400);

    const badMime = await request(app)
      .post('/api/visual-search')
      .attach('image', Buffer.from('plain text'), {
        filename: 'notes.txt',
        contentType: 'text/plain',
      });
    expect(badMime.status).toBe(400);
  });
});
