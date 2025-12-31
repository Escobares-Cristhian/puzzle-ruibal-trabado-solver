FROM python:3.11-slim

# Install ffmpeg for video generation
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# solver.py is in repo root
COPY solver.py /app/solver.py

# Ensure output folder exists at runtime before running the solver.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set environment variable for matplotlib backend
ENV MPLBACKEND=Agg

# Run the solver script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "solver.py"]
