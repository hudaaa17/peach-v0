FROM python:3.11-slim

# Install Node.js 22 (needed to run scip-python)
RUN apt-get update && apt-get install -y curl ca-certificates git \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Java + Joern (pinned to the version measured in the J1 spike)
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre-headless unzip \
    && rm -rf /var/lib/apt/lists/*
ARG JOERN_VERSION=v4.0.653
RUN curl -fL -o /tmp/joern-cli.zip "https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli-linux-x86_64.zip" \
    && unzip -q /tmp/joern-cli.zip -d /opt/joern \
    && rm /tmp/joern-cli.zip
ENV PATH="/opt/joern/joern-cli:${PATH}"

# Install the SCIP indexers
RUN npm install -g @sourcegraph/scip-python@0.6.6 @sourcegraph/scip-typescript

RUN pip install --no-cache-dir "ast-grep-cli==0.45.3" && ast-grep --version

RUN pip install pyyaml

# Install the scip CLI (reads index.scip files)
ARG SCIP_VERSION=v0.10.0
RUN curl -fsSL https://github.com/sourcegraph/scip/releases/download/${SCIP_VERSION}/scip-linux-amd64.tar.gz \
    | tar -xz -C /usr/local/bin scip

WORKDIR /app

# Install Python packages for the whole project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]

