import multer from 'multer';
import { env } from '../config/env.js';

const ALLOWED_MIMES = new Set([
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/webp',
  'image/gif',
  'image/bmp',
]);

export class UploadValidationError extends Error {
  statusCode: number;

  constructor(message: string, statusCode = 400) {
    super(message);
    this.statusCode = statusCode;
  }
}

export function createImageUploadMiddleware(maxBytes = env.MAX_IMAGE_BYTES) {
  return multer({
    storage: multer.memoryStorage(),
    limits: {
      fileSize: maxBytes,
      files: 1,
    },
    fileFilter: (_req, file, cb) => {
      if (!ALLOWED_MIMES.has(file.mimetype.toLowerCase())) {
        cb(new UploadValidationError(`Unsupported image type: ${file.mimetype}`));
        return;
      }
      cb(null, true);
    },
  });
}

export const uploadSingleImage = createImageUploadMiddleware();
