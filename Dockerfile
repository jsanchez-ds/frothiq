# FrothIQ FastAPI inference container.
#
# Build:  docker build -t frothiq-api .
# Run:    docker run -p 8000:8000 -v $(pwd)/mlruns:/app/mlruns frothiq-api
#
# Uses python:3.11-slim for a small image. PyTorch is excluded from the runtime
# install because the FastAPI service only needs LightGBM at inference. If you
# also want to serve the LSTM, change `--extra-index-url` to install torch CPU.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for LightGBM and pyarrow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml LICENSE README.md ./
COPY src ./src

# Install only inference deps. We exclude pyspark, jupyter, etc. to keep the image small.
RUN pip install \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2" \
        "pandas>=2.2" "numpy>=1.26" "scipy>=1.13" "scikit-learn>=1.5" \
        "lightgbm>=4.3" "mlflow>=2.14" "joblib" \
    && pip install --no-deps -e .

EXPOSE 8000

CMD ["uvicorn", "frothiq.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
