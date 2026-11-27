import { Router } from 'express';
import { z } from 'zod';
import { env } from '../config/env.js';
import { getClicksCollection } from '../db/mongo.js';

const ClickSchema = z.object({
  queryId: z.string().min(1),
  productId: z.string().min(1),
  imageUrl: z.string().min(1),
  rank: z.number().int().nonnegative(),
  score: z.number().min(0).max(1),
  modelVersion: z.string().optional(),
  filters: z
    .object({
      category: z.string().optional(),
      shape: z.string().optional(),
      color: z.string().optional(),
      metal: z.string().optional(),
      minPrice: z.number().optional(),
      maxPrice: z.number().optional(),
      inStock: z.boolean().optional(),
    })
    .optional(),
});

export function createClicksRouter(): Router {
  const router = Router();

  router.post('/clicks', async (req, res, next) => {
    try {
      const parsed = ClickSchema.parse(req.body);
      await getClicksCollection().insertOne({
        ...parsed,
        modelVersion: parsed.modelVersion ?? env.MODEL_VERSION,
        createdAt: new Date(),
      });
      res.status(201).json({ recorded: true });
    } catch (err) {
      next(err);
    }
  });

  return router;
}
