import { Collection, Db, MongoClient } from 'mongodb';
import { env } from '../config/env.js';
import type { ProductImageDoc, SearchClickDoc } from '../types/search.js';

let client: MongoClient | null = null;
let database: Db | null = null;

export async function connectMongo(uri = env.MONGODB_URI, dbName = env.MONGODB_DB): Promise<Db> {
  if (database) {
    return database;
  }
  client = new MongoClient(uri);
  await client.connect();
  database = client.db(dbName);
  await ensureIndexes(database);
  return database;
}

export async function ensureIndexes(db: Db): Promise<void> {
  const productImages = db.collection<ProductImageDoc>(env.MONGODB_COLLECTION);
  await productImages.createIndex(
    { productId: 1, imageUrl: 1, modelVersion: 1 },
    { unique: true, name: 'uniq_product_image_model' },
  );
  await productImages.createIndex(
    { 'attributes.category': 1, 'attributes.shape': 1, 'attributes.color': 1, inStock: 1, price: 1 },
    { name: 'idx_filter_facets' },
  );
  await productImages.createIndex({ modelVersion: 1 }, { name: 'idx_model_version' });

  const clicks = db.collection<SearchClickDoc>(env.CLICKS_COLLECTION);
  await clicks.createIndex({ queryId: 1, productId: 1 }, { name: 'idx_click_query' });
  await clicks.createIndex({ createdAt: -1 }, { name: 'idx_click_created' });
}

export function getDb(): Db {
  if (!database) {
    throw new Error('MongoDB is not connected. Call connectMongo() first.');
  }
  return database;
}

export function getProductImagesCollection(db: Db = getDb()): Collection<ProductImageDoc> {
  return db.collection<ProductImageDoc>(env.MONGODB_COLLECTION);
}

export function getClicksCollection(db: Db = getDb()): Collection<SearchClickDoc> {
  return db.collection<SearchClickDoc>(env.CLICKS_COLLECTION);
}

export async function closeMongo(): Promise<void> {
  if (client) {
    await client.close();
    client = null;
    database = null;
  }
}
