# Dockerfile
FROM python:3.11-slim

# Install system dependencies (SSH for SCP transfer)
RUN apt-get update && apt-get install -y openssh-client && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies from CLAUDE.md
RUN pip install --no-cache-dir \
    yfinance pandas numpy alpaca-py requests ollama GoogleNews

# Copy the entire trading agent repository into the container
COPY . /app/

# Make the shell script executable
RUN chmod +x /app/run_scout.sh

# Keep the container alive in the background so we can trigger it via cron or exec
CMD ["tail", "-f", "/dev/null"]
