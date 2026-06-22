# Deploying exopipe

`exopipe` ships as **one container image with two runtime shapes**:

| Shape | Cloud Run resource | Entrypoint | Use it for |
| --- | --- | --- | --- |
| **Dashboard** | **Service** (binds `$PORT`) | `dashboard` (default) | The interactive Streamlit candidate browser. |
| **Sector-scale batch** | **Job** | `run` / `job` -> `exopipe run` | Processing a directory of FITS / a CSV through the full pipeline. |

The same image (`Dockerfile`) backs both; `docker/entrypoint.sh` dispatches on
the first argument (default `dashboard`).

## When to use Cloud Run vs. a VM / GCP Batch

Cloud Run covers the common cases: an always-available dashboard service and
on-demand sector batches (up to **8 vCPU / 32 GiB** per task, 60-min default /
24-h max task timeout). Reach for a **Compute Engine instance** or **GCP Batch**
instead when a single task needs:

- a **GPU** (e.g. the deep-learning CNN classifier in the `dl` extra),
- **> 8 vCPU or > 32 GiB RAM** for one task, or
- an **always-on / long-lived** worker (Cloud Run Jobs are run-to-completion).

## Prerequisites (one-time GCP setup)

```bash
# Pick your values.
export GCP_PROJECT=my-project
export GCP_REGION=us-central1
export AR_REPO=exopipe

gcloud config set project "$GCP_PROJECT"

# Enable the APIs.
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# Create the Artifact Registry Docker repo (once).
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$GCP_REGION" \
  --description="exopipe container images"

export IMAGE="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/exopipe:latest"
```

## Build & push the image

Either let Cloud Build do it server-side:

```bash
gcloud builds submit --tag "$IMAGE" .
```

…or build locally and push:

```bash
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
docker build -t "$IMAGE" .
docker push "$IMAGE"
```

### The image is CPU-only by design

The default `Dockerfile` installs `".[science,ml,app]"` — the rules + XGBoost
ensemble, the Streamlit dashboard, and the TESS data stack. It deliberately
**excludes the `dl` (torch) extra**, and `xgboost` is pinned `<2.1` because
xgboost ≥ 2.1 declares a hard `nvidia-nccl-cu12` dependency that would bake a
multi-hundred-MB CUDA library into the image. The ensemble is fully functional
without the optional CNN, so the shipped image stays lean and CPU-only.

If you specifically want the optional CNN classifier, add the **CPU** torch
wheel — do **not** add CUDA to this image (use a GPU VM / GCP Batch for GPU
work, per the table above):

```bash
# In a derived image or at runtime, after the base install:
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```

## Deploy the dashboard (Cloud Run **service**)

```bash
gcloud run deploy exopipe-dashboard \
  --image "$IMAGE" \
  --region "$GCP_REGION" \
  --port 8080 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --session-affinity
```

`--session-affinity` pins each browser to one instance, which Streamlit's
websocket connection needs. The service starts in `dashboard` mode and renders
the bundled example catalog immediately. To point it at your own precomputed
results, set environment variables on the service:

```bash
gcloud run services update exopipe-dashboard \
  --region "$GCP_REGION" \
  --set-env-vars EXOPIPE_CATALOG=/data/catalog.csv,EXOPIPE_FIGDIR=/data/vetting_sheets
```

## Run a sector batch (Cloud Run **Job**)

Create the job once (note `--args run,...` selects the batch entrypoint):

```bash
gcloud run jobs create exopipe-batch \
  --image "$IMAGE" \
  --region "$GCP_REGION" \
  --args run,--input,/data/sector_lightcurves,--out,/data/runs/sector \
  --memory 4Gi \
  --cpu 4 \
  --task-timeout 3600 \
  --max-retries 1
```

Execute it (and re-execute whenever you want to process another input):

```bash
gcloud run jobs execute exopipe-batch --region "$GCP_REGION"
```

After pushing a new image, update the job to the new digest before executing:

```bash
gcloud run jobs update exopipe-batch --image "$IMAGE" --region "$GCP_REGION"
```

> Cloud Run Jobs have ephemeral local disk. Mount a GCS bucket (via
> `--add-volume`/`--add-volume-mount`) or read/write `gs://` paths so the input
> light curves and the output catalog survive the task.

## CI/CD

- **`.github/workflows/ci.yml`** runs on every push / PR (ruff + `pytest`).
- **`.github/workflows/deploy.yml`** is **manual** (`workflow_dispatch`) and uses
  Workload Identity Federation. Set these in
  *Settings -> Secrets and variables -> Actions* before triggering it:
  - Secrets: `GCP_WIF_PROVIDER`, `GCP_SERVICE_ACCOUNT`
  - Variables: `GCP_PROJECT`, `GCP_REGION`, `AR_REPO`

## Local docker

```bash
# Dashboard on http://localhost:8080
docker build -t exopipe .
docker run -p 8080:8080 exopipe

# Batch run against a local input (mount it in):
docker run -v "$PWD/data:/data" exopipe run --input /data/lightcurves --out /data/runs/local

# Offline synthetic end-to-end demo:
docker run -v "$PWD/out:/app/runs" exopipe demo --figures
```
