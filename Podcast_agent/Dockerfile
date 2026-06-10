FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p podcasts

# Expose port
EXPOSE 8002

# Run podcast agent
CMD ["python", "Podcast_agent/fastapi_podcast.py"]
