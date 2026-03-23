FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install pip dependencies explicitly supporting wheel structure
RUN pip install --no-cache-dir numpy matplotlib numba scipy scikit-opt pybind11 pandas seaborn joblib scikit-learn

# Build production shared library
RUN pip install .

ENTRYPOINT ["python", "runner.py"]
