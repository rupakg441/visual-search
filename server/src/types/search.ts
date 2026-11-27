export interface ProductAttributes {
  category?: string;
  shape?: string;
  color?: string;
  metal?: string;
  carat?: number | string;
  [key: string]: string | number | boolean | undefined;
}

export interface ProductImageDoc {
  productId: string;
  title?: string;
  imageUrl: string;
  localPath?: string;
  publicImageUrl?: string;
  productUrl?: string;
  embedding: number[];
  secondaryEmbedding?: number[];
  attributes: ProductAttributes;
  price: number;
  inStock: boolean;
  modelVersion: string;
  createdAt: Date;
  updatedAt?: Date;
}

export interface PredictedAttribute {
  value: string;
  confidence: number;
}

export type PredictedAttributes = Record<string, PredictedAttribute>;

export interface MlEmbedResponse {
  embedding: number[];
  dims: number;
  modelVersion: string;
  secondaryEmbedding?: number[] | null;
  secondaryDims?: number | null;
  predictedAttributes?: PredictedAttributes | null;
}

export interface SearchFilters {
  category?: string;
  shape?: string;
  color?: string;
  metal?: string;
  minPrice?: number;
  maxPrice?: number;
  inStock?: boolean;
  modelVersion?: string;
}

export interface ImageCandidate {
  productId: string;
  title?: string;
  imageUrl: string;
  localPath?: string;
  publicImageUrl?: string;
  productUrl?: string;
  attributes: ProductAttributes;
  price: number;
  inStock: boolean;
  modelVersion: string;
  vectorScore: number;
  secondaryEmbedding?: number[];
}

export interface RankedProductResult {
  productId: string;
  title?: string;
  imageUrl: string;
  publicImageUrl?: string;
  productUrl?: string;
  matchedImagesCount: number;
  attributes: ProductAttributes;
  price: number;
  inStock: boolean;
  modelVersion: string;
  similarity: number;
  vectorScore: number;
  attributeBoost: number;
  rerankScore: number;
}

export interface SearchClickDoc {
  queryId: string;
  productId: string;
  imageUrl: string;
  rank: number;
  score: number;
  filters?: SearchFilters;
  modelVersion: string;
  createdAt: Date;
}
