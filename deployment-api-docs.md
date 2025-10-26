# Documentation API et Déploiement

## 🚀 API Server Flask

### `src/deepsynth/inference/api_server.py`

```python
"""
Flask API server for DeepSynth (DeepSeek-OCR) summarization service
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
import uuid
from datetime import datetime
import logging
from typing import Dict, Any

from infer import DeepSynthSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize model (global to avoid reloading)
MODEL_PATH = os.getenv('MODEL_PATH', './deepsynth-summarizer')
summarizer = None

def init_model():
    """Initialize the summarization model"""
    global summarizer
    try:
        summarizer = DeepSynthSummarizer(MODEL_PATH)
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.before_first_request
def setup():
    init_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': summarizer is not None,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/summarize/text', methods=['POST'])
def summarize_text():
    """Summarize text input"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
            
        text = data['text']
        max_length = data.get('max_length', 128)
        temperature = data.get('temperature', 0.7)
        
        if len(text.strip()) == 0:
            return jsonify({'error': 'Empty text provided'}), 400
            
        # Generate summary
        summary = summarizer.summarize_text(
            text,
            max_length=max_length,
            temperature=temperature
        )
        
        # Calculate metrics
        compression_ratio = len(text) / len(summary) if summary else 0
        
        response = {
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': round(compression_ratio, 2),
            'parameters': {
                'max_length': max_length,
                'temperature': temperature
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in text summarization: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/summarize/file', methods=['POST'])
def summarize_file():
    """Summarize uploaded text file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Get parameters
        max_length = int(request.form.get('max_length', 128))
        temperature = float(request.form.get('temperature', 0.7))
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        try:
            # Read file content
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Generate summary
            summary = summarizer.summarize_text(
                text,
                max_length=max_length,
                temperature=temperature
            )
            
            compression_ratio = len(text) / len(summary) if summary else 0
            
            response = {
                'summary': summary,
                'filename': filename,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': round(compression_ratio, 2),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error in file summarization: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/summarize/image', methods=['POST'])
def summarize_image():
    """Summarize document image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        # Get parameters
        max_length = int(request.form.get('max_length', 128))
        temperature = float(request.form.get('temperature', 0.7))
        
        # Save image temporarily
        filename = secure_filename(image.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{filename}")
        image.save(temp_path)
        
        try:
            # Generate summary
            summary = summarizer.summarize_image(
                temp_path,
                max_length=max_length,
                temperature=temperature
            )
            
            response = {
                'summary': summary,
                'filename': filename,
                'summary_length': len(summary),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error in image summarization: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch/summarize', methods=['POST'])
def batch_summarize():
    """Batch text summarization"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400
            
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
            
        max_length = data.get('max_length', 128)
        temperature = data.get('temperature', 0.7)
        
        # Limit batch size
        if len(texts) > 10:
            return jsonify({'error': 'Maximum 10 texts per batch'}), 400
            
        # Generate summaries
        summaries = summarizer.batch_summarize(
            texts,
            max_length=max_length,
            temperature=temperature
        )
        
        # Prepare response
        results = []
        total_original = 0
        total_summary = 0
        
        for i, (text, summary) in enumerate(zip(texts, summaries)):
            original_len = len(text)
            summary_len = len(summary)
            total_original += original_len
            total_summary += summary_len
            
            results.append({
                'index': i,
                'summary': summary,
                'original_length': original_len,
                'summary_length': summary_len,
                'compression_ratio': round(original_len / summary_len, 2) if summary_len > 0 else 0
            })
        
        response = {
            'results': results,
            'batch_size': len(texts),
            'total_compression_ratio': round(total_original / total_summary, 2) if total_summary > 0 else 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch summarization: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--model-path', default='./deepsynth-summarizer')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    # Set model path
    MODEL_PATH = args.model_path
    
    print(f"🚀 Starting DeepSeek-OCR API server...")
    print(f"📍 Host: {args.host}:{args.port}")
    print(f"📂 Model: {MODEL_PATH}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
```

## 🐳 Docker Deployment

### `Dockerfile`

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    fonts-liberation \
    fonts-dejavu-core \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attention (may take time)
RUN pip install flash-attn --no-build-isolation

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models
ENV FLASK_APP=deepsynth.inference.api_server
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:5000/health || exit 1

# Start server
CMD ["python", "-m", "deepsynth.inference.api_server", "--host", "0.0.0.0", "--port", "5000"]
```

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  deepsynth-summarizer:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./temp:/app/temp
    environment:
      - MODEL_PATH=/app/models
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - deepsynth-summarizer
    restart: unless-stopped

volumes:
  models:
  temp:
```

### `nginx.conf`

```nginx
events {
    worker_connections 1024;
}

http {
    upstream deepsynth_backend {
        server deepsynth-summarizer:5000;
    }

    server {
        listen 80;
        server_name localhost;

        # Increase upload size for document files
        client_max_body_size 20M;

        # Proxy to API server
        location / {
            proxy_pass http://deepsynth_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings for long processing
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 300s;
        }

        # Health check
        location /health {
            proxy_pass http://deepsynth_backend/health;
            proxy_set_header Host $host;
        }
    }
}
```

## 📖 Client Usage Examples

### Python Client

```python
import requests
import json

class DeepSynthSummarizerClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
    
    def summarize_text(self, text: str, max_length: int = 128, temperature: float = 0.7):
        """Summarize text via API"""
        response = requests.post(
            f"{self.base_url}/summarize/text",
            json={
                "text": text,
                "max_length": max_length,
                "temperature": temperature
            }
        )
        return response.json()
    
    def summarize_file(self, file_path: str, max_length: int = 128, temperature: float = 0.7):
        """Summarize file via API"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'max_length': max_length,
                'temperature': temperature
            }
            response = requests.post(
                f"{self.base_url}/summarize/file",
                files=files,
                data=data
            )
        return response.json()
    
    def summarize_image(self, image_path: str, max_length: int = 128, temperature: float = 0.7):
        """Summarize document image via API"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'max_length': max_length,
                'temperature': temperature
            }
            response = requests.post(
                f"{self.base_url}/summarize/image",
                files=files,
                data=data
            )
        return response.json()
    
    def batch_summarize(self, texts: list, max_length: int = 128, temperature: float = 0.7):
        """Batch summarize texts via API"""
        response = requests.post(
            f"{self.base_url}/batch/summarize",
            json={
                "texts": texts,
                "max_length": max_length,
                "temperature": temperature
            }
        )
        return response.json()

# Usage example
if __name__ == "__main__":
    client = DeepSynthSummarizerClient()
    
    # Test text summarization
    text = """
    This is a long article about artificial intelligence and machine learning.
    It discusses various aspects of neural networks, deep learning, and their applications
    in natural language processing and computer vision.
    """
    
    result = client.summarize_text(text)
    print("Summary:", result['summary'])
    print("Compression ratio:", result['compression_ratio'])
```

### cURL Examples

```bash
# Test health check
curl -X GET http://localhost:5000/health

# Summarize text
curl -X POST http://localhost:5000/summarize/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "max_length": 128,
    "temperature": 0.7
  }'

# Summarize file
curl -X POST http://localhost:5000/summarize/file \
  -F "file=@document.txt" \
  -F "max_length=128" \
  -F "temperature=0.7"

# Summarize image
curl -X POST http://localhost:5000/summarize/image \
  -F "image=@document.png" \
  -F "max_length=128" \
  -F "temperature=0.7"

# Batch summarization
curl -X POST http://localhost:5000/batch/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Text 1...", "Text 2...", "Text 3..."],
    "max_length": 128,
    "temperature": 0.7
  }'
```

## ⚡ Performance Optimization

### `scripts/optimize_model.py`

```python
"""
Model optimization script for production deployment
"""

import torch
from transformers import AutoModel, AutoTokenizer
import argparse
import os

def optimize_model(model_path: str, output_path: str):
    """Optimize model for production"""
    print(f"🔧 Loading model from {model_path}")
    
    # Load model
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Apply optimizations
    print("⚡ Applying optimizations...")
    
    # 1. Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="max-autotune")
        print("✅ Model compiled with torch.compile")
    
    # 2. Export to TorchScript (optional)
    try:
        # Note: This may not work for all models
        scripted_model = torch.jit.script(model)
        script_path = os.path.join(output_path, "model_scripted.pt")
        scripted_model.save(script_path)
        print(f"✅ TorchScript model saved to {script_path}")
    except Exception as e:
        print(f"⚠️  TorchScript export failed: {e}")
    
    # 3. Save optimized model
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Optimized model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    
    args = parser.parse_args()
    optimize_model(args.model_path, args.output_path)
```

## 📊 Monitoring & Logging

### `monitoring/monitor.py`

```python
"""
Monitoring script for API server
"""

import time
import psutil
import GPUtil
import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIMonitor:
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        
    def get_system_metrics(self):
        """Get system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
        except:
            gpu_metrics = []
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'gpu_metrics': gpu_metrics
        }
    
    def check_api_health(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds(),
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_time': None,
                'status_code': None
            }
    
    def run_monitoring(self, interval: int = 30):
        """Run continuous monitoring"""
        logger.info(f"🔍 Starting monitoring (interval: {interval}s)")
        
        while True:
            try:
                # Get metrics
                system_metrics = self.get_system_metrics()
                api_health = self.check_api_health()
                
                # Combine metrics
                metrics = {
                    'system': system_metrics,
                    'api': api_health
                }
                
                # Log metrics
                logger.info(f"📊 Metrics: {json.dumps(metrics, indent=2)}")
                
                # Alert on high resource usage
                if system_metrics['cpu_percent'] > 80:
                    logger.warning(f"⚠️  High CPU usage: {system_metrics['cpu_percent']:.1f}%")
                
                if system_metrics['memory_percent'] > 90:
                    logger.warning(f"⚠️  High memory usage: {system_metrics['memory_percent']:.1f}%")
                
                for gpu in system_metrics['gpu_metrics']:
                    if gpu['load'] > 90:
                        logger.warning(f"⚠️  High GPU load: {gpu['load']:.1f}% on {gpu['name']}")
                
                if api_health['status'] != 'healthy':
                    logger.error(f"❌ API unhealthy: {api_health}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped")
                break
            except Exception as e:
                logger.error(f"💥 Monitoring error: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:5000")
    parser.add_argument("--interval", type=int, default=30)
    
    args = parser.parse_args()
    
    monitor = APIMonitor(args.api_url)
    monitor.run_monitoring(args.interval)
```

## 🚀 Deployment Commands

```bash
# 1. Build Docker image
docker build -t deepsynth-summarizer .

# 2. Run with Docker Compose
docker-compose up -d

# 3. Scale service
docker-compose up -d --scale deepsynth-summarizer=3

# 4. Monitor logs
docker-compose logs -f deepsynth-summarizer

# 5. Update service
docker-compose build deepsynth-summarizer
docker-compose up -d deepsynth-summarizer

# 6. Backup model
docker run --rm -v $(pwd)/models:/backup deepsynth-summarizer \
  tar czf /backup/model-$(date +%Y%m%d).tar.gz /app/models

# 7. Load test
ab -n 100 -c 10 -p test_data.json \
   -T "application/json" \
   http://localhost:5000/summarize/text
```

---

*Cette documentation complète fournit tous les éléments nécessaires pour déployer et maintenir le service DeepSeek-OCR en production, avec monitoring, optimisations et haute disponibilité.*