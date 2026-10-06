FROM python:3.11-slim

# Install Node.js 22 (needed to run scip-python)
RUN apt-get update && apt-get install -y curl ca-certificates git \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install the SCIP indexers
RUN npm install -g @sourcegraph/scip-python@0.6.6 @sourcegraph/scip-typescript

# Install the scip CLI (reads index.scip files)
ARG SCIP_VERSION=v0.10.0
RUN curl -fsSL https://github.com/sourcegraph/scip/releases/download/${SCIP_VERSION}/scip-linux-amd64.tar.gz \
    | tar -xz -C /usr/local/bin scip

WORKDIR /app

# Install Python packages for the whole project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]

