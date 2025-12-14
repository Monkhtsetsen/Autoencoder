FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy project files into container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Run Flask app with Gunicorn (production-safe)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--preload", "--timeout", "120"]
