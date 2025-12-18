# MedCraft AI

MedCraft is an enterprise-grade AI platform for medical analysis, featuring:
- **Titans Memory**: Long-term neural memory with surprise-based updates (MAC/MAG/MAL).
- **NanoVLM**: Vision-Language Model with Perceiver Resampler and TinyLlama (QLoRA).
- **Hybrid Vision**: YOLOv8 + ViT for robust detection and classification.
- **MLOps**: Full Airflow pipelines for data, training, and evaluation.

**[DATA SETUP INSTRUCTIONS](DATA_SETUP.md)** - Read this first to populate datasets.

## 🚀 Quick Start

### Local Development
```bash
# Start all services (CPU)
make dev

# Start with GPU support (Triton, QLoRA)
make dev_gpu
```

### Evaluation
Run the full evaluation suite to generate `eval/INVESTOR_REPORT.pdf`:
```bash
make eval_all
```

## 🏗 Architecture

### Titans Memory (`services/titans_memory`)
Implements the Titans architecture (arXiv:2501.00663) with:
- **Neural Memory**: Gradient-based updates on surprise loss.
- **Variants**: Memory as Context (MAC), Gating (MAG), Layer (MAL).
- **Persistence**: Redis-backed state storage.

### NanoVLM (`services/nanovlm`)
- **Vision**: Frozen CLIP ViT.
- **Connector**: Trainable Perceiver Resampler.
- **LLM**: TinyLlama-1.1B (QLoRA supported).
- **Inference**: Structured JSON output.

### Hybrid Vision (`services/vision_yolo`)
- **Detection**: YOLOv8n.
- **Classification**: ViT-Base on ROIs.
- **Export**: ONNX/TensorRT ready.

## 🚢 Deployment

### One-Shot Azure Deployment
The platform is deployed via GitHub Actions (`.github/workflows/deploy.yml`).
1. Set `AZURE_CREDENTIALS` in GitHub Secrets.
2. Push to `main` branch.
3. The pipeline will:
   - Provision Infrastructure (AKS, ACR, Postgres, Redis) via Terraform.
   - Build and Push Docker Images.
   - Deploy via Helm.
   - Run Smoke Tests.

### Manual Deployment
```bash
# Build images
make build

# Deploy to Kubernetes
helm upgrade --install medcraft ops/helm/medcraft
```
