import { enqueueProductImageIndex } from '../server/src/queues/indexQueue.js';
import { closeMongo, connectMongo } from '../server/src/db/mongo.js';
import type { IndexProductImageJob } from '../server/src/queues/indexProcessor.js';

const sample: IndexProductImageJob[] = [
  { productId: 'sample-gold-ring', title: 'Gold solitaire ring', imageUrl: 'https://images.unsplash.com/photo-1605100804763-247f67b3557e?w=800', attributes: { category: 'rings', color: 'gold', shape: 'round' }, price: 1200, inStock: true },
  { productId: 'sample-diamond-ring', title: 'Diamond engagement ring', imageUrl: 'https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=800', attributes: { category: 'rings', color: 'white', shape: 'round' }, price: 2400, inStock: true },
];
async function main() {
  await connectMongo();
  try { for (const item of sample) console.log(await enqueueProductImageIndex(item)); }
  finally { await closeMongo(); }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
