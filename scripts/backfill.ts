import fs from 'node:fs/promises';
import path from 'node:path';
import { Queue } from 'bullmq';
import dotenv from 'dotenv';
import { Redis } from 'ioredis';
import { MongoClient } from 'mongodb';

interface ScrapedCatalogItem {
  productId: string;
  title: string;
  imageUrl: string;
  localPath: string;
  publicImageUrl: string;
  productUrl: string;
  attributes: {
    category: string;
    shape?: string;
    metal?: string;
    color?: string;
    carat?: string;
  };
  price: number;
  inStock: boolean;
}

dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

interface CliOptions {
  inputPath: string;
  checkpointPath: string;
  batchSize: number;
  useQueue: boolean;
  resetCheckpoint: boolean;
  modelVersion: string;
}

function parseArgs(): CliOptions {
  const args = process.argv.slice(2);
  const opts: CliOptions = {
    inputPath: path.resolve(process.cwd(), '../data/catalog.json'),
    checkpointPath: path.resolve(process.cwd(), '../data/backfill-checkpoint.json'),
    batchSize: 16,
    useQueue: (process.env.QUEUE_ENABLED ?? 'false').toLowerCase() === 'true',
    resetCheckpoint: false,
    modelVersion: process.env.MODEL_VERSION ?? 'openclip-vit-b32-v1',
  };

  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--input' && args[i + 1]) opts.inputPath = path.resolve(args[++i]);
    else if (args[i] === '--checkpoint' && args[i + 1]) opts.checkpointPath = path.resolve(args[++i]);
    else if (args[i] === '--batch-size' && args[i + 1]) opts.batchSize = Math.max(1, Number(args[++i]) || 16);
    else if (args[i] === '--use-queue') opts.useQueue = true;
    else if (args[i] === '--direct') opts.useQueue = false;
    else if (args[i] === '--reset') opts.resetCheckpoint = true;
    else if (args[i] === '--model-version' && args[i + 1]) opts.modelVersion = args[++i];
  }
  return opts;
}

async function loadCheckpoint(filePath: string, modelVersion: string): Promise<Set<string>> {
  try {
    const raw = await fs.readFile(filePath, 'utf8');
    const parsed = JSON.parse(raw) as { modelVersion?: string; completedKeys?: string[] };
    if (parsed.modelVersion === modelVersion && Array.isArray(parsed.completedKeys)) {
      return new Set(parsed.completedKeys);
    }
  } catch {
    // No checkpoint yet
  }
  return new Set();
}

async function saveCheckpoint(filePath: string, modelVersion: string, keys: Set<string>): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(
    filePath,
    JSON.stringify({ modelVersion, updatedAt: new Date().toISOString(), completedKeys: Array.from(keys) }, null, 2),
    'utf8',
  );
}

async function withRetry<T>(fn: () => Promise<T>, attempts = 3, baseDelayMs = 1000): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt < attempts) {
        await new Promise((r) => setTimeout(r, baseDelayMs * attempt));
      }
    }
  }
  throw lastError;
}

async function embedBatchViaMl(
  mlUrl: string,
  items: ScrapedCatalogItem[],
  imageDir: string,
): Promise<
  Array<{
    embedding: number[];
    secondaryEmbedding?: number[] | null;
    modelVersion: string;
  }>
> {
  const form = new FormData();
  for (const item of items) {
    const absPath = path.resolve(imageDir, item.localPath);
    const bytes = await fs.readFile(absPath);
    const ext = path.extname(absPath).toLowerCase();
    const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg';
    form.append('files', new Blob([new Uint8Array(bytes)], { type: mime }), path.basename(absPath));
  }

  const res = await fetch(`${mlUrl}/embed/batch`, { method: 'POST', body: form });
  if (!res.ok) {
    throw new Error(`ML /embed/batch failed (${res.status}): ${await res.text()}`);
  }
  const body = (await res.json()) as {
    modelVersion: string;
    embeddings: Array<{
      embedding: number[];
      secondaryEmbedding?: number[] | null;
      modelVersion: string;
    }>;
  };
  return body.embeddings;
}

async function main(): Promise<void> {
  const opts = parseArgs();
  const imageDir = path.resolve(process.cwd(), process.env.LOCAL_IMAGE_DIR ?? '../data/images');
  const rawCatalog = await fs.readFile(opts.inputPath, 'utf8');
  const items = JSON.parse(rawCatalog) as ScrapedCatalogItem[];

  const completed = opts.resetCheckpoint
    ? new Set<string>()
    : await loadCheckpoint(opts.checkpointPath, opts.modelVersion);

  const itemKey = (item: ScrapedCatalogItem) => `${item.productId}::${item.imageUrl}`;
  const pending = items.filter((item) => !completed.has(itemKey(item)));

  console.log(
    `[backfill] total=${items.length} already_done=${completed.size} pending=${pending.length} mode=${opts.useQueue ? 'bullmq' : 'direct'} modelVersion=${opts.modelVersion}`,
  );

  if (pending.length === 0) {
    console.log('[backfill] nothing to do.');
    return;
  }

  if (opts.useQueue) {
    const redis = new Redis(process.env.REDIS_URL ?? 'redis://localhost:6379', {
      maxRetriesPerRequest: null,
      enableReadyCheck: false,
    });
    const queue = new Queue(process.env.INDEX_QUEUE_NAME ?? 'product-image-index', {
      connection: redis,
      defaultJobOptions: {
        attempts: 3,
        backoff: { type: 'exponential', delay: 2000 },
        removeOnComplete: 1000,
      },
    });

    for (let i = 0; i < pending.length; i += opts.batchSize) {
      const slice = pending.slice(i, i + opts.batchSize);
      await queue.addBulk(
        slice.map((item) => ({
          name: 'embed-product-image',
          data: { ...item, modelVersion: opts.modelVersion },
          opts: {
            jobId: `${item.productId}__${Buffer.from(item.imageUrl).toString('base64url')}__${opts.modelVersion}`,
          },
        })),
      );
      for (const item of slice) completed.add(itemKey(item));
      await saveCheckpoint(opts.checkpointPath, opts.modelVersion, completed);
      console.log(`[backfill] enqueued ${Math.min(i + slice.length, pending.length)}/${pending.length}`);
    }

    await queue.close();
    await redis.quit();
    return;
  }

  const mongoUri = process.env.MONGODB_URI ?? 'mongodb://localhost:27017';
  const dbName = process.env.MONGODB_DB ?? 'visual_search';
  const colName = process.env.MONGODB_COLLECTION ?? 'product_images';
  const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';

  const client = new MongoClient(mongoUri);
  await client.connect();

  try {
    const col = client.db(dbName).collection(colName);
    await col.createIndex(
      { productId: 1, imageUrl: 1, modelVersion: 1 },
      {
        unique: true,
        name: 'uniq_product_image_model_valid',
        partialFilterExpression: {
          productId: { $type: 'string', $gt: '' },
          imageUrl: { $type: 'string', $gt: '' },
          modelVersion: { $type: 'string', $gt: '' },
        },
      },
    );

    const startedAt = Date.now();
    for (let i = 0; i < pending.length; i += opts.batchSize) {
      const batch = pending.slice(i, i + opts.batchSize);
      const embeddings = await withRetry(() => embedBatchViaMl(mlUrl, batch, imageDir));
      const now = new Date();

      const ops = batch.map((item, idx) => {
        const emb = embeddings[idx];
        const ver = emb.modelVersion ?? opts.modelVersion;
        return {
          updateOne: {
            filter: {
              productId: item.productId,
              imageUrl: item.imageUrl,
              modelVersion: ver,
            },
            update: {
              $set: {
                productId: item.productId,
                title: item.title,
                imageUrl: item.imageUrl,
                localPath: item.localPath,
                publicImageUrl: item.publicImageUrl,
                productUrl: item.productUrl,
                embedding: emb.embedding,
                ...(emb.secondaryEmbedding ? { secondaryEmbedding: emb.secondaryEmbedding } : {}),
                attributes: item.attributes,
                price: item.price,
                inStock: item.inStock,
                modelVersion: ver,
                updatedAt: now,
              },
              $setOnInsert: { createdAt: now },
            },
            upsert: true,
          },
        };
      });

      await col.bulkWrite(ops, { ordered: false });
      for (const item of batch) completed.add(itemKey(item));
      await saveCheckpoint(opts.checkpointPath, opts.modelVersion, completed);

      const done = Math.min(i + batch.length, pending.length);
      const elapsedSec = Math.max(0.1, (Date.now() - startedAt) / 1000);
      const rate = (done / elapsedSec).toFixed(1);
      console.log(`[backfill] indexed ${done}/${pending.length} (${completed.size}/${items.length} total) | ${rate} img/s`);
    }

    const totalDocs = await col.countDocuments({ modelVersion: opts.modelVersion });
    console.log(`[backfill] complete. MongoDB ${dbName}.${colName} count for ${opts.modelVersion}: ${totalDocs}`);
  } finally {
    await client.close();
  }
}

main().catch((err) => {
  console.error('[backfill] fatal error:', err);
  process.exit(1);
});
