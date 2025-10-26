# DeepSynth Dataset Generator & Trainer - Docker UI

A Docker-based web interface for generating datasets and training models with the DeepSeek-OCR framework. Features include:

> _Repository note_: Docker examples still reference the legacy slug `deepseek-synthesia` until the GitHub rename is complete.

- **Resumable Jobs**: Automatically resume interrupted dataset generation or training
- **Incremental Datasets**: Add new samples without duplicates
- **Real-time Monitoring**: Track progress and view statistics
- **HuggingFace Integration**: Seamless upload to HuggingFace Hub
- **GPU Support**: CUDA-enabled for fast training

---

## Quick Start

### Prerequisites

1. **Docker & Docker Compose** installed
2. **NVIDIA Docker** (for GPU support)
3. **HuggingFace Account** with API token

### Setup

1. **Clone the repository**:
```bash
cd deepseek-synthesia
```

2. **Configure environment variables**:
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Required variables:
```bash
HF_TOKEN=hf_your_token_here
HF_USERNAME=your-username
SECRET_KEY=your-secret-key-for-flask
```

3. **Build and run**:
```bash
# Build the Docker image
docker-compose build

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f
```

4. **Access the UI**:
Open your browser to: **http://localhost:5000**

---

## Using the Web Interface

### 1. Dataset Generation

Navigate to the **"Generate Dataset"** tab and fill in:

#### Source Dataset
- **HuggingFace Dataset**: e.g., `ccdv/cnn_dailymail`
- **Subset**: e.g., `3.0.0` (optional)
- **Split**: `train`, `validation`, or `test`

#### Field Mapping
- **Text Field**: Column containing the text to convert to images (e.g., `article`)
- **Summary Field**: Column containing summaries (e.g., `highlights`)

#### Output Configuration
- **Output Directory**: Where to save images (e.g., `./generated_images`)
- **Max Samples**: Limit number of samples (optional, leave empty for all)

#### HuggingFace Hub
- **HF Username**: Your HuggingFace username
- **Dataset Name**: Name for the generated dataset
- **Private Dataset**: Check to make dataset private

Click **"Generate Dataset"** to start!

### 2. Model Training

Navigate to the **"Train Model"** tab and configure:

#### Dataset
- **HuggingFace Dataset Repo**: `username/dataset-name` (from previous step)

#### Model Configuration
- **Model Name**: `deepseek-ai/DeepSeek-OCR` (default)
- **Output Directory**: Where to save trained model

#### Training Parameters
- **Batch Size**: 2 (adjust based on GPU memory)
- **Epochs**: 1 or more
- **Learning Rate**: `2e-5` (default)
- **Max Length**: 512 tokens
- **Mixed Precision**: `bf16` (recommended) or `fp16`
- **Gradient Accumulation**: 4 steps (default)

#### Push to Hub
- Check **"Push model to Hub"** to upload trained model
- Provide **Model ID**: `username/model-name`

Click **"Start Training"** to begin!

### 3. Monitor Jobs

Navigate to the **"Monitor Jobs"** tab to:

- **View all jobs** with real-time progress
- **Filter** by job type (Dataset or Training)
- **Resume** paused or failed jobs
- **Pause** running jobs
- **View details** for each job
- **Delete** completed jobs

Auto-refreshes every 5 seconds when active.

---

## Job Resumption & Deduplication

### How It Works

1. **State Persistence**: All job state is saved to `./web_ui/state/`
2. **Hash Tracking**: Each processed sample is hashed (SHA256)
3. **Duplicate Detection**: New samples are checked against processed hashes
4. **Incremental Upload**: Datasets are uploaded to HuggingFace every 100 samples
5. **Resume from Checkpoint**: Jobs can be resumed from where they stopped

### Resume a Job

If a job fails or is interrupted:

1. Go to **"Monitor Jobs"** tab
2. Find the job with status `failed` or `paused`
3. Click **"Resume"**

The job will:
- Load existing state
- Skip already processed samples
- Continue from where it stopped
- Append new samples to existing HuggingFace dataset

---

## Docker Commands

### Basic Operations

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

### Check Status

```bash
# Check running containers
docker ps

# Check resource usage
docker stats deepsynth-dataset-generator

# Access container shell
docker exec -it deepsynth-dataset-generator bash
```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker rmi deepsynth-ui
```

---

## Directory Structure

```
deepseek-synthesia/
├── web_ui/
│   ├── app.py                    # Flask backend
│   ├── state_manager.py          # Job state management
│   ├── dataset_generator.py      # Dataset generation logic
│   ├── templates/
│   │   └── index.html            # Web UI
│   ├── static/
│   │   ├── styles.css            # Styling
│   │   └── script.js             # Frontend logic
│   └── state/                    # Job state (persisted)
│       ├── jobs.json
│       └── hashes/
├── generated_images/             # Generated PNG images (volume)
├── trained_model/                # Trained models (volume)
├── logs/                         # Application logs (volume)
├── Dockerfile
├── docker-compose.yml
└── .env                          # Environment variables
```

---

## API Endpoints

The Flask backend provides REST API endpoints:

### Jobs
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/<job_id>` - Get job details
- `GET /api/jobs/<job_id>/progress` - Get job progress
- `DELETE /api/jobs/<job_id>` - Delete job

### Dataset Generation
- `POST /api/dataset/generate` - Start dataset generation

### Model Training
- `POST /api/model/train` - Start model training

### Job Control
- `POST /api/jobs/<job_id>/resume` - Resume job
- `POST /api/jobs/<job_id>/pause` - Pause job

### Health Check
- `GET /api/health` - Check service health

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If error, install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Port Already in Use

```bash
# Change port in docker-compose.yml
ports:
  - "8080:5000"  # Change 5000 to another port
```

### Out of Memory

Reduce batch size in training configuration:
- **Batch Size**: 1
- **Gradient Accumulation**: 8 or more

### HuggingFace Authentication Failed

```bash
# Login to HuggingFace
docker exec -it deepsynth-dataset-generator bash
huggingface-cli login

# Or update HF_TOKEN in .env
```

### Job Stuck

1. Check logs: `docker-compose logs -f`
2. Restart service: `docker-compose restart`
3. If persists, delete job and recreate

---

## Advanced Configuration

### Custom Dockerfile

Edit `Dockerfile` to:
- Change CUDA version
- Add additional dependencies
- Modify Python version

### Environment Variables

Add to `.env`:

```bash
# Custom port
PORT=8080

# Increase workers
WORKERS=4

# Custom CUDA devices
CUDA_VISIBLE_DEVICES=0,1
```

### Volumes

Add custom volumes in `docker-compose.yml`:

```yaml
volumes:
  - ./custom_data:/app/custom_data
  - ./checkpoints:/app/checkpoints
```

---

## Performance Tips

1. **GPU Memory**: Monitor with `nvidia-smi` and adjust batch size
2. **Disk Space**: Monitor image directory size
3. **Network**: HuggingFace uploads can be slow for large datasets
4. **Concurrent Jobs**: Run multiple dataset generations in parallel
5. **Checkpoint Frequency**: Adjust save frequency in `dataset_generator.py`

---

## Security Notes

1. **Change SECRET_KEY** in production
2. **Use HTTPS** with reverse proxy (nginx/traefik)
3. **Restrict access** with firewall rules
4. **Secure HF_TOKEN** - never commit to git
5. **Use private datasets** for sensitive data

---

## Example Workflow

### Complete Pipeline

1. **Generate Dataset**:
   - Source: `ccdv/cnn_dailymail`
   - Max samples: 1000
   - Output: `./generated_images`
   - Upload to: `username/cnn-vision-1k`

2. **Monitor Progress**:
   - Check "Monitor Jobs" tab
   - Wait for completion (or resume if interrupted)

3. **Train Model**:
   - Dataset: `username/cnn-vision-1k`
   - Epochs: 1
   - Push to Hub: `username/deepsynth-cnn-model`

4. **Use Trained Model**:
```bash
# Inside container
python -m deepsynth.inference.infer \
  --model ./trained_model \
  --text "Your document text here"
```

---

## Support

- **Issues**: Report at GitHub issues
- **Documentation**: See [main README](../README.md)
- **Logs**: Check `./logs/` directory

---

## License

Same as parent project.
