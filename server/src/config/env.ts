import path from 'node:path';
import dotenv from 'dotenv';
import { z } from 'zod';

// Load root .env when running from /server or monorepo root
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });
dotenv.config();

const EnvSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  PORT: z.coerce.number().int().positive().default(4000),
  CLIENT_ORIGIN: z.string().default('http://localhost:5173'),
  LOCAL_IMAGE_DIR: z.string().default(path.resolve(process.cwd(), '../data/images')),

  MONGODB_URI: z.string().default('mongodb://localhost:27017'),
  MONGODB_DB: z.string().default('visual_search'),
  MONGODB_COLLECTION: z.string().default('product_images'),
  CLICKS_COLLECTION: z.string().default('search_clicks'),

  VECTOR_SEARCH_MODE: z.enum(['atlas', 'exact']).default('exact'),
  ATLAS_VECTOR_INDEX: z.string().default('product_images_vector_index'),

  REDIS_URL: z.string().default('redis://localhost:6379'),
  INDEX_QUEUE_NAME: z.string().default('product-image-index'),
  QUEUE_ENABLED: z
    .string()
    .default('false')
    .transform((v) => v.toLowerCase() === 'true'),

  ML_SERVICE_URL: z.string().url().default('http://localhost:8000'),
  ML_TIMEOUT_MS: z.coerce.number().int().positive().default(15000),
  MODEL_VERSION: z.string().default('openclip-vit-b32-v1'),
  DINOV2_WEIGHT: z.coerce.number().min(0).max(1).default(0.35),

  RATE_LIMIT_WINDOW_MS: z.coerce.number().int().positive().default(60000),
  RATE_LIMIT_MAX: z.coerce.number().int().positive().default(60),
  MAX_IMAGE_BYTES: z.coerce.number().int().positive().default(10 * 1024 * 1024),

  ENABLE_HYBRID_RANKING: z
    .string()
    .default('true')
    .transform((v) => v.toLowerCase() === 'true'),
  HYBRID_ATTR_BOOST: z.coerce.number().min(0).max(1).default(0.12),
  ENABLE_RERANKING: z
    .string()
    .default('true')
    .transform((v) => v.toLowerCase() === 'true'),
  RERANK_CANDIDATES: z.coerce.number().int().positive().default(50),
  TOP_RESULTS: z.coerce.number().int().positive().default(20),
});

export type AppEnv = z.infer<typeof EnvSchema>;

export function loadEnv(): AppEnv {
  return EnvSchema.parse(process.env);
}

export const env = loadEnv();
