import { Worker } from 'bullmq';
import { env } from '../config/env.js';
import { closeMongo, connectMongo, getProductImagesCollection } from '../db/mongo.js';
import { createRedisConnection } from './connection.js';
import { processIndexJob, type IndexProductImageJob } from './indexProcessor.js';

async function startWorker(): Promise<void> {
  await connectMongo();
  const collection = getProductImagesCollection();
  const connection = createRedisConnection();

  const worker = new Worker<IndexProductImageJob>(
    env.INDEX_QUEUE_NAME,
    async (job) => {
      await processIndexJob(job.data, collection);
    },
    {
      connection,
      concurrency: 4,
    },
  );

  worker.on('completed', (job) => {
    console.log(`[worker] indexed product=${job.data.productId} image=${job.data.imageUrl}`);
  });

  worker.on('failed', (job, err) => {
    console.error(`[worker] failed job=${job?.id} product=${job?.data.productId}:`, err.message);
  });

  const shutdown = async () => {
    console.log('[worker] shutting down...');
    await worker.close();
    await connection.quit();
    await closeMongo();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  console.log(`[worker] listening on queue "${env.INDEX_QUEUE_NAME}"`);
}

startWorker().catch((err) => {
  console.error('[worker] fatal startup error:', err);
  process.exit(1);
});
