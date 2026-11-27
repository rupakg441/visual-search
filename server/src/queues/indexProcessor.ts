import fs from 'node:fs/promises';
import path from 'node:path';
import type { Collection } from 'mongodb';
import { env } from '../config/env.js';
import { mlClient, type MlClient } from '../services/mlClient.js';
import type { ProductAttributes, ProductImageDoc } from '../types/search.js';

export interface IndexProductImageJob {
  productId: string;
  title?: string;
  imageUrl: string;
  localPath?: string;
  publicImageUrl?: string;
  productUrl?: string;
  attributes: ProductAttributes;
  price: number;
  inStock: boolean;
  modelVersion?: string;
}

async function readImageBytes(job: IndexProductImageJob): Promise<{ buffer: Buffer; mimeType: string }> {
  if (job.localPath) {
    const root = path.resolve(env.LOCAL_IMAGE_DIR);
    const resolved = path.resolve(root, job.localPath);
    const relative = path.relative(root, resolved);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      throw new Error('localPath must resolve inside LOCAL_IMAGE_DIR.');
    }
    const stat = await fs.stat(resolved);
    if (stat.size > env.MAX_IMAGE_BYTES) throw new Error(`Catalog image exceeds ${env.MAX_IMAGE_BYTES} bytes.`);
    const buffer = await fs.readFile(resolved);
    const ext = path.extname(resolved).toLowerCase();
    const mimeType =
      ext === '.png'
        ? 'image/png'
        : ext === '.webp'
          ? 'image/webp'
          : 'image/jpeg';
    return { buffer, mimeType };
  }

  const res = await fetch(job.imageUrl, { signal: AbortSignal.timeout(30_000) });
  if (!res.ok) {
    throw new Error(`Failed to fetch catalog image ${job.imageUrl} (${res.status})`);
  }
  const contentLength = Number(res.headers.get('content-length') ?? 0);
  if (contentLength > env.MAX_IMAGE_BYTES) throw new Error(`Catalog image exceeds ${env.MAX_IMAGE_BYTES} bytes.`);
  const reader = res.body?.getReader();
  if (!reader) throw new Error('Catalog image response had no body.');
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > env.MAX_IMAGE_BYTES) {
      await reader.cancel();
      throw new Error(`Catalog image exceeds ${env.MAX_IMAGE_BYTES} bytes.`);
    }
    chunks.push(value);
  }
  const mimeType = res.headers.get('content-type') ?? 'image/jpeg';
  return { buffer: Buffer.concat(chunks), mimeType };
}

/**
 * Idempotent embedding & upsert processor shared by BullMQ worker, direct API reindex,
 * and catalog backfill CLI.
 */
export async function processIndexJob(
  job: IndexProductImageJob,
  collection: Collection<ProductImageDoc>,
  client: MlClient = mlClient,
): Promise<ProductImageDoc> {
  const { buffer, mimeType } = await readImageBytes(job);
  const filename = path.basename(job.localPath ?? job.imageUrl.split('?')[0]) || 'product.jpg';
  const embedRes = await client.embedImage(buffer, filename, mimeType);
  const modelVersion = job.modelVersion ?? embedRes.modelVersion ?? env.MODEL_VERSION;

  const now = new Date();
  const docFields: Omit<ProductImageDoc, 'createdAt'> = {
    productId: job.productId,
    title: job.title,
    imageUrl: job.imageUrl,
    localPath: job.localPath,
    publicImageUrl: job.publicImageUrl,
    productUrl: job.productUrl,
    embedding: embedRes.embedding,
    ...(embedRes.secondaryEmbedding ? { secondaryEmbedding: embedRes.secondaryEmbedding } : {}),
    attributes: job.attributes,
    price: job.price,
    inStock: job.inStock,
    modelVersion,
    updatedAt: now,
  };

  await collection.updateOne(
    {
      productId: job.productId,
      imageUrl: job.imageUrl,
      modelVersion,
    },
    {
      $set: docFields,
      $setOnInsert: { createdAt: now },
    },
    { upsert: true },
  );

  return { ...docFields, createdAt: now };
}
