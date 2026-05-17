FROM python:3.11-slim

# System deps: Tkinter (for pt_hub.py), Xvfb (headless X11), build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-tk \
        tk-dev \
        xvfb \
        x11-utils \
        libx11-6 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source (secrets are NOT copied — see .dockerignore)
COPY . .

# Runtime directories that should be volume-mounted for persistence
RUN mkdir -p hub_data backtest_cache backtest_results optimizer_results

# Default: Streamlit dashboard (no display required)
# Override CMD in docker-compose.yml per service
EXPOSE 8501
CMD ["streamlit", "run", "pt_dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
