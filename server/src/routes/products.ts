import { Router } from 'express';
import { z } from 'zod';
import { env } from '../config/env.js';
import { getProductImagesCollection } from '../db/mongo.js';
import { HttpError } from '../middleware/errorHandler.js';
import { enqueueProductImageIndex } from '../queues/indexQueue.js';

const ProductImageInputSchema = z.object({
  productId: z.string().min(1),
  title: z.string().optional(),
  imageUrl: z.string().url(),
  localPath: z.string().optional(),
  publicImageUrl: z.string().optional(),
  productUrl: z.string().url().optional(),
  attributes: z.record(z.union([z.string(), z.number(), z.boolean()])).default({}),
  price: z.number().nonnegative().default(0),
  inStock: z.boolean().default(true),
  modelVersion: z.string().optional(),
});

export function createProductsRouter(): Router {
  const router = Router();

  router.get('/catalog-image', async (req, res, next) => {
    try {
      const rawUrl = typeof req.query.url === 'string' ? req.query.url : '';
      let imageUrl: URL;
      try {
        imageUrl = new URL(rawUrl);
      } catch {
        throw new HttpError('A valid catalog image URL is required.', 400);
      }

      if (
        imageUrl.protocol !== 'https:' ||
        imageUrl.hostname !== 'cdn.diamondforest.com' ||
        imageUrl.port !== '' ||
        !imageUrl.pathname.startsWith('/Products/')
      ) {
        throw new HttpError('Only Diamond Forest catalog image URLs are allowed.', 400);
      }

      const upstream = await fetch(imageUrl, {
        signal: AbortSignal.timeout(20_000),
        redirect: 'error',
      });
      if (!upstream.ok) {
        throw new HttpError(`Catalog image request failed (${upstream.status}).`, 502);
      }

      const contentType = upstream.headers.get('content-type')?.split(';')[0].trim().toLowerCase();
      if (!contentType?.startsWith('image/')) {
        throw new HttpError('The catalog URL did not return an image.', 502);
      }

      const declaredLength = Number(upstream.headers.get('content-length') ?? 0);
      if (declaredLength > env.MAX_IMAGE_BYTES) {
        throw new HttpError('Catalog image exceeds the upload size limit.', 413);
      }

      const reader = upstream.body?.getReader();
      if (!reader) throw new HttpError('Catalog image response had no body.', 502);
      const chunks: Uint8Array[] = [];
      let total = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        total += value.byteLength;
        if (total > env.MAX_IMAGE_BYTES) {
          await reader.cancel();
          throw new HttpError('Catalog image exceeds the upload size limit.', 413);
        }
        chunks.push(value);
      }

      res
        .status(200)
        .set('Content-Type', contentType)
        .set('Cache-Control', 'public, max-age=3600')
        .send(Buffer.concat(chunks));
    } catch (err) {
      next(err);
    }
  });

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
