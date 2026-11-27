import { Router } from 'express';
import { getProductImagesCollection } from '../db/mongo.js';
import { ProductImageInputSchema } from '../db/models/product-image.js';
import { enqueueProductImageIndex } from '../queues/indexQueue.js';

export function createProductsRouter(): Router {
  const router = Router();

  router.get('/products', async (req, res, next) => {
    try {
      const limit = Math.min(Number(req.query.limit ?? 24) || 24, 100);
      const col = getProductImagesCollection();
      const docs = await col
        .find({}, { projection: { embedding: 0, secondaryEmbedding: 0 } })
        .sort({ createdAt: -1 })
        .limit(limit)
        .toArray();

      const [categories, shapes, metals, colors] = await Promise.all([
        col.distinct('attributes.category'),
        col.distinct('attributes.shape'),
        col.distinct('attributes.metal'),
        col.distinct('attributes.color'),
      ]);

      res.json({
        count: docs.length,
        facets: {
          categories: categories.filter(Boolean),
          shapes: shapes.filter(Boolean),
          metals: metals.filter(Boolean),
          colors: colors.filter(Boolean),
        },
        items: docs,
      });
    } catch (err) {
      next(err);
    }
  });

  router.post('/products/reindex', async (req, res, next) => {
    try {
      const payload = ProductImageInputSchema.parse(req.body);
      const result = await enqueueProductImageIndex(payload);
      res.status(result.queued ? 202 : 200).json({
        status: result.queued ? 'queued' : 'indexed',
        ...result,
      });
    } catch (err) {
      next(err);
    }
  });

  return router;
}
