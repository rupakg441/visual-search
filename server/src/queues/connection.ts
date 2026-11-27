import { Redis } from 'ioredis';
import { env } from '../config/env.js';

export function createRedisConnection(redisUrl = env.REDIS_URL): Redis {
  return new Redis(redisUrl, {
    maxRetriesPerRequest: null,
    enableReadyCheck: false,
  });
}
