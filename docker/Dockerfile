# docker build -t llm-benchmarking -f docker/Dockerfile .
# docker run --rm -v $(pwd)/output:/app/output -v $(pwd)/logs:/app/logs -v $(pwd)/src:/app/src llm-benchmarking

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for Python packages with scientific dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3 \
    python3-pip \
    git \
    zsh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip==24.0
RUN python3 -m pip install -r requirements.txt

# Copy the rest of the code
COPY src/ /app/src/
COPY data/ /app/data/
COPY logs/ /app/logs/

# Set working directory to the src directory
WORKDIR /app/

# Command to run when container starts
CMD ["/bin/zsh"]
# CMD ["python3", "-m", "src.regression.regressor"]