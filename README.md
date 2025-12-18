# MedCraft: Multimodal Medical AI Platform

MedCraft is a production-grade, end-to-end multimodal medical AI platform designed for clinical decision support. It features specialized modules for Parkinson's screening, Chest X-Ray analysis, Retina tracking, and Pathology WSI analysis, all augmented by a custom NanoVLM reasoning engine and Titans long-term memory.

## Architecture

- **Frontend**: Next.js (TypeScript) with a premium dark theme.
- **API Gateway**: FastAPI with Auth, RBAC, and Redis Streams.
- **Vision Service**: YOLOv8 + ViT for localization and classification.
- **NanoVLM Service**: Custom VLM architecture for grounded clinical QA.
- **Titans Memory Service**: MAC, MAG, and MAL layers for context management.
- **MLOps**: Airflow, MLflow, DVC, and Triton Inference Server.
- **Infrastructure**: Terraform (Azure) and Helm (AKS).

## Deployment

### One-Shot Azure Deployment (Recommended)

This project is configured for automated "one-shot" deployment to Azure via GitHub Actions.

**Prerequisites:**
1.  **Azure Account**: Active subscription.
2.  **GitHub Repository**: Push this code to a GitHub repo.

**Steps:**

1.  **Create Azure Service Principal**:
    ```bash
    az ad sp create-for-rbac --name "medcraft-cicd" --role contributor --scopes /subscriptions/<SUBSCRIPTION_ID> --sdk-auth
    ```
    Copy the JSON output.

2.  **Configure GitHub Secrets**:
    Go to `Settings > Secrets and variables > Actions` in your repo and add:
    - `AZURE_CREDENTIALS`: Paste the JSON from step 1.

3.  **Trigger Deployment**:
    - Push to `main` branch OR
    - Go to "Actions" tab > "MedCraft One-Shot Deploy" > "Run workflow".

**What happens automatically:**
- Terraform provisions AKS, ACR, Postgres, Redis.
- Docker images are built and pushed to ACR.
- Helm charts deploy all services to AKS.
- The platform becomes live at the AKS LoadBalancer IP.

### Local Development (CPU)
1.  **Environment Setup**:
    ```bash
    cp .env.example .env
    # Edit .env if needed (defaults work for local)
    ```
2.  **Run Stack**:
    ```bash
    make dev
    ```
    Access UI at `http://localhost:3000`.

### Local Development (GPU)
```bash
make dev_gpu
```

## Project Structure
- `/apps/web`: Next.js frontend.
- `/services/api`: FastAPI gateway.
- `/services/vision_yolo`: Vision service.
- `/services/nanovlm`: Reasoning service.
- `/services/titans_memory`: Memory service.
- `/ops/airflow`: ML orchestration.
- `/ops/terraform`: Azure IaC.
- `/ops/helm`: K8s manifests.
- `/data`: DVC-managed data.
- `/eval`: Evaluation pipeline.

## License
Proprietary - MedCraft AI
