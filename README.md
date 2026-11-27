# Jewelry Visual Search (MERN + FastAPI ML)

Production-ready visual product search monorepo tailored for **Diamond Forest** jewelry (`visual_search.product_images`). Users upload or capture an image of a ring, band, necklace, earring, or bracelet and retrieve the most visually similar catalog products with attribute/price filtering, hybrid ranking, and top-50 reranking.

---

## Architecture

```text
React (Vite + TS) ──▶ Express API (TS) ──▶ FastAPI ML Service (Python)
                           │                      │
                           ▼                      ▼
                    MongoDB (Atlas          OpenCLIP / SigLIP / DINOv2
                    $vectorSearch or        + YOLOv8 Crop + rembg
                    exact cosine)           + Optional TTA & ONNX
                           ▲
                           │
                 BullMQ + Redis Worker / Resumable Backfill CLI
```

### Repository Layout
- `client/` — React 18 + Vite + TypeScript UI (drag-and-drop, file picker, mobile camera capture, interactive canvas cropper, filters, similarity badges, skeletons, click-through logging).
- `server/` — Node.js 20 + Express + TypeScript API, BullMQ + Redis worker, hybrid attribute ranker, top-50 reranker, Multer upload validation, rate limiting, and Vitest unit/integration tests.
- `ml-service/` — Python FastAPI microservice (`/health`, `/embed`, `/embed/batch`) with shared preprocessing (PIL validation, YOLOv8 object crop with full-image fallback, `rembg` neutral-white background removal, resize/normalize, optional TTA & ONNX Runtime).
- `scripts/` — Diamond Forest scraper (`scrape.ts`), resumable batch catalog backfill CLI (`backfill.ts`), evaluation harness (`evaluate.ts` with Recall@K, Precision@K, MRR, p95 latency & `--compare`), and PyTorch contrastive fine-tuning (`finetune/`).
- `server/src/db/atlas-vector-index.json` — MongoDB Atlas Vector Search index definition (`product_images_vector_index`).

---

## Prerequisites
- **Node.js 20+**
- **Python 3.11+** (or Docker Compose)
- **MongoDB** (Local `mongodb://localhost:27017` with `VECTOR_SEARCH_MODE=exact`, or MongoDB Atlas / `mongodb-atlas-local` with `VECTOR_SEARCH_MODE=atlas`)
- **Redis** (optional in local direct mode with `QUEUE_ENABLED=false`; required for BullMQ worker `QUEUE_ENABLED=true`)

---

## Quick Start (Local Development)

### 1. Configure Environment
```bash
cp .env.example .env
```
Key defaults for local MongoDB:
- `MONGODB_URI=mongodb://localhost:27017`
- `MONGODB_DB=visual_search`
- `MONGODB_COLLECTION=product_images`
- `VECTOR_SEARCH_MODE=exact` (set to `atlas` on MongoDB Atlas)

### 2. Start the Python ML Service
```bash
cd ml-service
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Start the Express Server (and optional BullMQ Worker)
```bash
cd server
npm install
npm run dev
# Optional when QUEUE_ENABLED=true and Redis is running:
npm run worker
```

### 4. Scrape & Index 1,000 Diamond Forest Product Images
```bash
cd scripts
npm install
# Scrape df-lr3529 + wedding-rings + Diamond Forest catalog up to 1,000 images locally
npm run scrape -- --limit 1000
# Embed & upsert idempotently into MongoDB visual_search.product_images
npm run backfill -- --batch-size 16
```
To enqueue via BullMQ + Redis instead of direct batching:
```bash
npm run backfill -- --use-queue --batch-size 32
```

### 5. Start the React Frontend
```bash
cd client
npm install
npm run dev
```
Open `http://localhost:5173`.

---

## Docker Compose (Full Stack)
```bash
cp .env.example .env
docker compose up --build
```
To include a local Atlas-compatible MongoDB container for `$vectorSearch`:
```bash
docker compose --profile local-atlas up --build
```

---

## MongoDB Atlas Vector Search Setup
Create a Vector Search index named `product_images_vector_index` on `visual_search.product_images` using [`server/src/db/atlas-vector-index.json`](server/src/db/atlas-vector-index.json):
- Similarity: `cosine`
- Path: `embedding` (`512` dims for OpenCLIP `ViT-B-32`; adjust if switching to `SigLIP` or `DINOv2`)
- Filter fields: `inStock`, `modelVersion`, `price`, `attributes.category`, `attributes.shape`, `attributes.color`, `attributes.metal`, `attributes.carat`

Then set `VECTOR_SEARCH_MODE=atlas` in `.env`.

---

## API Reference

### Express API (`http://localhost:4000`)
- `GET /api/health` — Returns MongoDB & ML service readiness.
- `POST /api/visual-search` (`multipart/form-data`)
  - `image`: image file (`<= 10 MB`, JPEG/PNG/WebP/GIF)
  - Optional form fields: `category`, `shape`, `color`, `metal`, `minPrice`, `maxPrice`, `inStock`, `exact` (`true`/`false`), `hybridRanking`, `rerank`, `tta`, `limit`
  - Returns `{ queryId, searchMode, modelVersion, latencyMs, totalCandidates, predictedAttributes, results[] }`
- `GET /api/products` — Returns indexed catalog preview & distinct filter facets.
- `POST /api/products/reindex` — Enqueues/processes a product image embedding update idempotently.
- `POST /api/clicks` — Records click-through feedback in `search_clicks`.

### Python ML Service (`http://localhost:8000`)
- `GET /health` — Model, dimensions, device, and preprocessing status.
- `POST /embed` — Single image file -> `{ embedding, dims, modelVersion, secondaryEmbedding?, predictedAttributes? }`
- `POST /embed/batch` — Multiple image files -> `{ embeddings[], modelVersion }`

---

## Evaluation & Fine-Tuning

### Run the Evaluation Harness
```bash
cd scripts
# Auto-generates data/eval-set.json from multi-view catalog products if not present
npm run evaluate -- --generate --max-queries 50

# Compare baseline vector search vs hybrid + rerank + TTA side-by-side
npm run evaluate -- --compare
```
Prints `recallAt5`, `recallAt10`, `precisionAt5`, `mrr`, `latencyP50Ms`, and `latencyP95Ms`.

### Fine-Tune on Catalog Pairs
```bash
cd scripts/finetune
python train_contrastive.py --catalog ../../data/catalog.json --image-dir ../../data/images --epochs 3 --output-version openclip-vit-b32-ft-v1
python export_model.py --checkpoint ../../models/finetuned/openclip-vit-b32-ft-v1.pt --onnx-out ../../models/onnx/ft-v1.onnx
```
Then set `MODEL_VERSION=openclip-vit-b32-ft-v1` and `FINETUNED_WEIGHTS_PATH=...` and re-run `npm run backfill -- --reset` for zero-downtime re-indexing.

---

## Tests
- **ML Service (pytest):**
  ```bash
  cd ml-service
  pytest -q
  ```
- **Server Route Unit + MongoDB Integration (Vitest):**
  ```bash
  cd server
  npm test
  ```
