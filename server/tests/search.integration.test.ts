import { describe, expect, it } from 'vitest';
import request from 'supertest';
import { MongoClient } from 'mongodb';
import { createApp } from '../src/app.js';
import type { MlClient } from '../src/services/mlClient.js';
import type { ProductImageDoc } from '../src/types/search.js';

const TEST_URI = process.env.MONGODB_URI ?? 'mongodb://localhost:27017';
const TEST_DB = 'visual_search_integration_test';

describe('Visual search integration (Express + MongoDB)', () => {
  it('indexes product images into MongoDB and retrieves the closest product via POST /api/visual-search', async () => {
    const client = new MongoClient(TEST_URI, { serverSelectionTimeoutMS: 3000 });
    try {
      await client.connect();
    } catch {
      // Skip cleanly if MongoDB is unavailable in CI without a service container
      return;
    }

    try {
      const db = client.db(TEST_DB);
      const col = db.collection<ProductImageDoc>('product_images');
      await col.deleteMany({});

      await col.insertMany([
        {
          productId: 'df-lr3529',
          title: 'Carmen Marquise Step Cut Ring',
          imageUrl: 'https://cdn.diamondforest.com/Products/Rings/LR3529/LR3529_DMQ_20/LR3529_DMQ_20_GOLD_1.webp',
          embedding: [0.98, 0.2, 0.0, 0.0],
          attributes: { category: 'engagement-rings', shape: 'marquise', metal: 'yellow-gold', color: 'gold' },
          price: 2450,
          inStock: true,
          modelVersion: 'integration-v1',
          createdAt: new Date(),
        },
        {
          productId: 'df-other',
          title: 'Solitaire Round Band',
          imageUrl: 'https://cdn.diamondforest.com/Products/Rings/LR5107/LR5107_BR_3/LR5107_BR_3_WHITE_1.webp',
          embedding: [0.0, 0.1, 0.99, 0.0],
          attributes: { category: 'engagement-rings', shape: 'round', metal: 'white-gold', color: 'white' },
          price: 1600,
          inStock: true,
          modelVersion: 'integration-v1',
          createdAt: new Date(),
        },
      ]);

      const fakeMl: MlClient = {
        embedImage: async () => ({
          embedding: [0.99, 0.14, 0.0, 0.0],
          dims: 4,
          modelVersion: 'integration-v1',
        }),
        health: async () => ({ status: 'ok', modelVersion: 'integration-v1' }),
      };

      const app = createApp({
        getCollection: () => col,
        mlClient: fakeMl,
        applyRateLimit: false,
        checkMongoHealth: async () => {
          await db.command({ ping: 1 });
          return true;
        },
      });

      const health = await request(app).get('/api/health');
      expect(health.status).toBe(200);
      expect(health.body.mongo).toBe('connected');

      const searchRes = await request(app)
        .post('/api/visual-search')
        .field('exact', 'true')
        .field('minPrice', '2000')
        .attach('image', Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
          filename: 'ring.png',
          contentType: 'image/png',
        });

      expect(searchRes.status).toBe(200);
      expect(searchRes.body.results).toHaveLength(1);
      expect(searchRes.body.results[0].productId).toBe('df-lr3529');
    } finally {
      await client.db(TEST_DB).dropDatabase().catch(() => undefined);
      await client.close();
    }
  });
});
