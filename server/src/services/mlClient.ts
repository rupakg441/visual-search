import { env } from '../config/env.js';
import { HttpError } from '../middleware/errorHandler.js';
import type { MlEmbedResponse } from '../types/search.js';

export interface EmbedRequestOptions {
  tta?: boolean;
  skipBgRemoval?: boolean;
  skipDetection?: boolean;
}

export interface MlClient {
  embedImage(
    buffer: Buffer,
    filename: string,
    mimeType: string,
    options?: EmbedRequestOptions,
  ): Promise<MlEmbedResponse>;
  health(): Promise<Record<string, unknown>>;
}

export function createMlClient(
  baseUrl = env.ML_SERVICE_URL,
  timeoutMs = env.ML_TIMEOUT_MS,
): MlClient {
  async function requestWithTimeout(url: string, init?: RequestInit): Promise<Response> {
    let lastError: unknown;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetch(url, { ...init, signal: controller.signal });
        if (response.status >= 500 && attempt < 2) {
          await response.body?.cancel();
          await new Promise((resolve) => setTimeout(resolve, 150 * (attempt + 1)));
          continue;
        }
        return response;
      } catch (err) {
        lastError = err;
        if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 150 * (attempt + 1)));
      } finally {
        clearTimeout(timer);
      }
    }
    if (lastError instanceof Error && lastError.name === 'AbortError') {
      throw new HttpError(`ML service timed out after ${timeoutMs} ms (3 attempts).`, 504);
    }
    const cause = lastError instanceof Error
      ? (lastError as Error & { cause?: unknown }).cause
      : undefined;
    const causeText = cause instanceof Error
      ? `${cause.message}${'code' in cause && typeof cause.code === 'string' ? ` (${cause.code})` : ''}`
      : lastError instanceof Error ? lastError.message : 'unknown network error';
    const host = new URL(baseUrl).host;
    throw new HttpError(
      `ML embedding service at ${host} is unreachable after 3 attempts. Start the ML service or check ML_SERVICE_URL. Cause: ${causeText}`,
      502,
    );
  }

  return {
    async embedImage(buffer, filename, mimeType, options = {}) {
      const params = new URLSearchParams();
      if (options.tta !== undefined) params.set('tta', String(options.tta));
      if (options.skipBgRemoval) params.set('skip_bg_removal', 'true');
      if (options.skipDetection) params.set('skip_detection', 'true');

      const query = params.toString();
      const targetUrl = `${baseUrl}/embed${query ? `?${query}` : ''}`;

      const form = new FormData();
      const blob = new Blob([new Uint8Array(buffer)], { type: mimeType || 'image/jpeg' });
      form.append('file', blob, filename || 'query.jpg');

      const res = await requestWithTimeout(targetUrl, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const detail = await res.text();
        const status = res.status >= 400 && res.status < 500 ? 400 : 502;
        throw new HttpError(`ML service error (${res.status}): ${detail}`, status);
      }

      return (await res.json()) as MlEmbedResponse;
    },

    async health() {
      const res = await requestWithTimeout(`${baseUrl}/health`);
      if (!res.ok) {
        throw new HttpError(`ML service unhealthy (${res.status})`, 502);
      }
      return (await res.json()) as Record<string, unknown>;
    },
  };
}

export const mlClient = createMlClient();
