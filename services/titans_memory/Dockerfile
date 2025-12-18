FROM python:3.10-slim

WORKDIR /app

# Install torch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir fastapi uvicorn pydantic

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
