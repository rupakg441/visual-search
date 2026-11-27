import path from 'node:path';
import cors from 'cors';
import express from 'express';
import helmet from 'helmet';
import { env } from './config/env.js';
import { getDb } from './db/mongo.js';
import { errorHandler } from './middleware/errorHandler.js';
import { requestLogger } from './middleware/logger.js';
import { createClicksRouter } from './routes/clicks.js';
import { createProductsRouter } from './routes/products.js';
import { createVisualSearchRouter, type VisualSearchRouterDeps } from './routes/visualSearch.js';
import { mlClient as defaultMlClient } from './services/mlClient.js';

export interface AppOptions extends VisualSearchRouterDeps {
  localImageDir?: string;
  checkMongoHealth?: () => Promise<boolean>;
}

export function createApp(options: AppOptions = {}) {
  const app = express();
  const ml = options.mlClient ?? defaultMlClient;
  const localImageDir = options.localImageDir ?? env.LOCAL_IMAGE_DIR;

  app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
  app.use(cors({ origin: env.CLIENT_ORIGIN }));
  app.use(express.json({ limit: '2mb' }));
  app.use(requestLogger);

  // Serve locally downloaded catalog images cleanly
  app.use('/images', express.static(path.resolve(localImageDir)));

  app.get('/api/health', async (_req, res) => {
    let mongoOk = false;
    try {
      if (options.checkMongoHealth) {
        mongoOk = await options.checkMongoHealth();
      } else {
        await getDb().command({ ping: 1 });
        mongoOk = true;
      }
    } catch {
      mongoOk = false;
    }

    let mlHealth: Record<string, unknown> | null = null;
    try {
      mlHealth = await ml.health();
    } catch {
      mlHealth = null;
    }

    const healthy = mongoOk && mlHealth !== null;
    res.status(healthy ? 200 : 503).json({
      status: healthy ? 'ok' : 'degraded',
      mongo: mongoOk ? 'connected' : 'disconnected',
      vectorSearchMode: env.VECTOR_SEARCH_MODE,
      mlService: mlHealth ?? 'unreachable',
    });
  });

  app.use('/api', createVisualSearchRouter(options));
  app.use('/api', createProductsRouter());
  app.use('/api', createClicksRouter());

  app.use(errorHandler);
  return app;
}
