import fs from 'node:fs/promises';
import path from 'node:path';
import dotenv from 'dotenv';
import type { ScrapedCatalogItem } from './scrape.js';

dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

export interface EvalCase {
  queryImage: string;
  relevantProductIds: string[];
  filters?: Record<string, string | number | boolean>;
}

interface SearchConfig {
  name: string;
  fields: Record<string, string>;
}

interface EvalMetrics {
  configName: string;
  queries: number;
  recallAt5: number;
  recallAt10: number;
  precisionAt5: number;
  mrr: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
}

function parseCli() {
  const args = process.argv.slice(2);
  let evalPath = path.resolve(process.cwd(), '../data/eval-set.json');
  let apiUrl = process.env.EVAL_API_URL ?? `http://localhost:${process.env.PORT ?? 4000}/api/visual-search`;
  let compare = false;
  let generateFromCatalog = false;
  let maxQueries = 50;

  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--dataset' && args[i + 1]) evalPath = path.resolve(args[++i]);
    else if (args[i] === '--api-url' && args[i + 1]) apiUrl = args[++i];
    else if (args[i] === '--compare') compare = true;
    else if (args[i] === '--generate') generateFromCatalog = true;
    else if (args[i] === '--max-queries' && args[i + 1]) maxQueries = Number(args[++i]) || 50;
  }
  return { evalPath, apiUrl, compare, generateFromCatalog, maxQueries };
}

/**
 * Builds a realistic evaluation dataset from multi-image products in catalog.json:
 * selects one image per product as query and marks its exact productId as relevant.
 */
async function buildEvalDatasetFromCatalog(evalPath: string, maxQueries: number): Promise<EvalCase[]> {
  const catalogPath = path.resolve(process.cwd(), '../data/catalog.json');
  const imageDir = path.resolve(process.cwd(), process.env.LOCAL_IMAGE_DIR ?? '../data/images');
  const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8')) as ScrapedCatalogItem[];

  const byProduct = new Map<string, ScrapedCatalogItem[]>();
  for (const item of catalog) {
    const list = byProduct.get(item.productId) ?? [];
    list.push(item);
    byProduct.set(item.productId, list);
  }

  const cases: EvalCase[] = [];
  for (const [productId, imgs] of byProduct.entries()) {
    if (cases.length >= maxQueries) break;
    // Prefer products with >= 2 images so query can test cross-angle/metal retrieval
    const chosen = imgs.length > 1 ? imgs[imgs.length - 1] : imgs[0];
    cases.push({
      queryImage: path.resolve(imageDir, chosen.localPath),
      relevantProductIds: [productId],
    });
  }

  await fs.mkdir(path.dirname(evalPath), { recursive: true });
  await fs.writeFile(evalPath, JSON.stringify(cases, null, 2), 'utf8');
  console.log(`[eval] generated ${cases.length} evaluation cases at ${evalPath}`);
  return cases;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)];
}

async function runSingleEval(
  cases: EvalCase[],
  apiUrl: string,
  config: SearchConfig,
): Promise<EvalMetrics> {
  let recall5Sum = 0;
  let recall10Sum = 0;
  let prec5Sum = 0;
  let rrSum = 0;
  const latencies: number[] = [];

  for (const testCase of cases) {
    const imgBytes = await fs.readFile(testCase.queryImage);
    const ext = path.extname(testCase.queryImage).toLowerCase();
    const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg';

    const form = new FormData();
    form.append('image', new Blob([new Uint8Array(imgBytes)], { type: mime }), path.basename(testCase.queryImage));
    for (const [k, v] of Object.entries(config.fields)) {
      form.append(k, v);
    }
    if (testCase.filters) {
      for (const [k, v] of Object.entries(testCase.filters)) {
        form.append(k, String(v));
      }
    }

    const t0 = performance.now();
    const res = await fetch(apiUrl, { method: 'POST', body: form });
    const elapsed = performance.now() - t0;
    latencies.push(elapsed);

    if (!res.ok) {
      throw new Error(`Search failed for ${testCase.queryImage}: ${res.status} ${await res.text()}`);
    }

    const body = (await res.json()) as { results: Array<{ productId: string }> };
    const returnedIds = body.results.map((r) => r.productId);
    const relSet = new Set(testCase.relevantProductIds);

    const top5 = returnedIds.slice(0, 5);
    const top10 = returnedIds.slice(0, 10);

    const hits5 = top5.filter((id) => relSet.has(id)).length;
    const hits10 = top10.filter((id) => relSet.has(id)).length;

    recall5Sum += Math.min(1, hits5 / Math.max(1, relSet.size));
    recall10Sum += Math.min(1, hits10 / Math.max(1, relSet.size));
    prec5Sum += hits5 / 5;

    const firstRank = returnedIds.findIndex((id) => relSet.has(id));
    rrSum += firstRank >= 0 ? 1 / (firstRank + 1) : 0;
  }

  const n = Math.max(1, cases.length);
  return {
    configName: config.name,
    queries: cases.length,
    recallAt5: Number((recall5Sum / n).toFixed(4)),
    recallAt10: Number((recall10Sum / n).toFixed(4)),
    precisionAt5: Number((prec5Sum / n).toFixed(4)),
    mrr: Number((rrSum / n).toFixed(4)),
    latencyP50Ms: Math.round(percentile(latencies, 50)),
    latencyP95Ms: Math.round(percentile(latencies, 95)),
  };
}

async function main(): Promise<void> {
  const cli = parseCli();

  let cases: EvalCase[];
  if (cli.generateFromCatalog) {
    cases = await buildEvalDatasetFromCatalog(cli.evalPath, cli.maxQueries);
  } else {
    try {
      cases = JSON.parse(await fs.readFile(cli.evalPath, 'utf8')) as EvalCase[];
    } catch {
      console.log('[eval] eval dataset not found; generating from data/catalog.json...');
      cases = await buildEvalDatasetFromCatalog(cli.evalPath, cli.maxQueries);
    }
  }

  const configs: SearchConfig[] = cli.compare
    ? [
        {
          name: 'Baseline (Vector Only)',
          fields: { hybridRanking: 'false', rerank: 'false', tta: 'false' },
        },
        {
          name: 'Enhanced (Hybrid + Rerank + TTA)',
          fields: { hybridRanking: 'true', rerank: 'true', tta: 'true' },
        },
      ]
    : [
        {
          name: 'Default Pipeline',
          fields: { hybridRanking: 'true', rerank: 'true' },
        },
      ];

  const report: EvalMetrics[] = [];
  for (const cfg of configs) {
    console.log(`[eval] running "${cfg.name}" across ${cases.length} queries...`);
    report.push(await runSingleEval(cases, cli.apiUrl, cfg));
  }

  console.table(report);
}

main().catch((err) => {
  console.error('[eval] fatal error:', err);
  process.exit(1);
});
