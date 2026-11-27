import { Queue } from 'bullmq';
import { env } from '../config/env.js';
import { getProductImagesCollection } from '../db/mongo.js';
import { createRedisConnection } from './connection.js';
import { processIndexJob, type IndexProductImageJob } from './indexProcessor.js';

let queueInstance: Queue<IndexProductImageJob> | null = null;

export function getIndexQueue(): Queue<IndexProductImageJob> {
  if (!queueInstance) {
    queueInstance = new Queue<IndexProductImageJob>(env.INDEX_QUEUE_NAME, {
      connection: createRedisConnection(),
      defaultJobOptions: {
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 2000,
        },
        removeOnComplete: 500,
        removeOnFail: 500,
      },
    });
  }
  return queueInstance;
}

/**
 * Enqueues a product image embedding job via BullMQ when Redis queue is enabled,
 * or processes it directly in-process when QUEUE_ENABLED=false.
 */
export async function enqueueProductImageIndex(
  job: IndexProductImageJob,
): Promise<{ queued: boolean; jobId: string }> {
  const modelVersion = job.modelVersion ?? env.MODEL_VERSION;
  const deterministicJobId = `${job.productId}__${Buffer.from(job.imageUrl).toString('base64url')}__${modelVersion}`;

  if (!env.QUEUE_ENABLED) {
    await processIndexJob(job, getProductImagesCollection());
    return { queued: false, jobId: deterministicJobId };
  }

  const queue = getIndexQueue();
  await queue.add('embed-product-image', job, { jobId: deterministicJobId });
  return { queued: true, jobId: deterministicJobId };
}

export async function closeIndexQueue(): Promise<void> {
  if (queueInstance) {
    await queueInstance.close();
    queueInstance = null;
  }
}
