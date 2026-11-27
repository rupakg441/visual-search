import { z } from 'zod';

export const SearchClickInputSchema = z.object({
  queryId: z.string().min(1), productId: z.string().min(1), imageUrl: z.string().min(1),
  rank: z.number().int().nonnegative(), score: z.number().min(0).max(1), modelVersion: z.string().optional(),
});
export type SearchClickInput = z.infer<typeof SearchClickInputSchema>;
