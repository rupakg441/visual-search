import { env } from './config/env.js';
import { closeMongo, connectMongo } from './db/mongo.js';
import { closeIndexQueue } from './queues/indexQueue.js';
import { createApp } from './app.js';

async function main(): Promise<void> {
  await connectMongo();
  const app = createApp();

  const server = app.listen(env.PORT, () => {
    console.log(
      `[server] listening on http://localhost:${env.PORT} | db=${env.MONGODB_DB}.${env.MONGODB_COLLECTION} | mode=${env.VECTOR_SEARCH_MODE}`,
    );
  });

  const shutdown = async () => {
    console.log('[server] shutting down...');
    server.close();
    await closeIndexQueue();
    await closeMongo();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch((err) => {
  console.error('[server] failed to start:', err);
  process.exit(1);
});
