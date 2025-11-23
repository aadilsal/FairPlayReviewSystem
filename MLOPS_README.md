# MLOps Video Processing Pipeline - Complete Setup

This project integrates a video processing ML pipeline with MLflow model tracking on DagsHub, featuring a Streamlit frontend and FastAPI backend.

## 📋 Project Structure

```
FairPlayReviewSystem/
├── frontend/                    # Streamlit UI
│   ├── app.py                  # Video upload interface
│   └── requirements.txt
├── mlops/                      # MLOps pipeline
│   ├── config.py               # Configuration loader
│   ├── video_processor.py      # Video preprocessing
│   ├── model_manager.py        # MLflow model management
│   ├── inference.py            # Inference pipeline
│   ├── server.py               # FastAPI backend
│   ├── train_and_register_model.py  # Model registration script
│   ├── test_model_loading.py   # Model loading test
│   ├── utils.py                # Helper functions
│   └── requirements.txt
├── .env                        # Environment variables (DO NOT COMMIT)
├── .env.example                # Example environment config
└── test_mlflow_connection.py  # MLflow connectivity test
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- DagsHub account (free at https://dagshub.com/user/sign_up)
- Git

### Step 1: Create DagsHub Repository

1. **Go to DagsHub**: https://dagshub.com/repo/create
2. **Create new repository**:
   - Name: `FairPlayReviewSystem` (or your choice)
   - Visibility: Public/Private
3. **Enable MLflow** in the repository (Experiments tab)

### Step 2: Install Dependencies

```powershell
# Install mlops backend dependencies
pip install -r mlops/requirements.txt

# Install frontend dependencies
pip install -r frontend/requirements.txt
```

### Step 3: Configure Environment

Copy `.env.example` to `.env` and update with your credentials:

```env
# DagsHub & MLflow
DAGSHUB_USERNAME=aadilsal234
DAGSHUB_PAT=your-token-here
MLFLOW_TRACKING_URI=https://dagshub.com/aadilsal234/FairPlayReviewSystem.mlflow
MLFLOW_TRACKING_USERNAME=aadilsal234
MLFLOW_TRACKING_PASSWORD=your-token-here

# Model Configuration
MLFLOW_MODEL_NAME=tf-object-detection
MODEL_VERSION=Staging

# Paths
UPLOAD_DIR=./mlops/data/uploads
RESULTS_DIR=./mlops/data/results
MAX_VIDEO_SIZE=500

# Inference Settings
DEVICE=cpu

# Backend URL
BACKEND_URL=http://localhost:8000
```

**Important**: Replace `your-token-here` with your actual DagsHub Personal Access Token.

### Step 4: Test MLflow Connection

```powershell
python test_mlflow_connection.py
```

Expected output:

```
✓ MLflow connection successful!
Run ID: abc123...
View at: https://dagshub.com/aadilsal234/FairPlayReviewSystem.mlflow
```

### Step 5: Train and Register Model

```powershell
python mlops/train_and_register_model.py
```

This will:

1. Load pre-trained TensorFlow EfficientDet model from TensorFlow Hub
2. Register it to MLflow Model Registry on DagsHub
3. Transition to "Staging" stage

Expected output:

```
✓ Model loaded successfully
✓ Model logged to MLflow (Run ID: ...)
✓ Model registered: tf-object-detection, Version: 1, Stage: Staging
```

### Step 6: Verify Model Loading

```powershell
python mlops/test_model_loading.py
```

### Step 7: Start Backend Server

```powershell
cd mlops
python server.py
```

Server will start at: `http://localhost:8000`

### Step 8: Start Frontend (New Terminal)

```powershell
cd frontend
streamlit run app.py
```

Frontend will open at: `http://localhost:8501`

## 🧪 Testing the Complete Pipeline

### Option 1: Using Streamlit UI

1. Open `http://localhost:8501` in your browser
2. Upload a video file (.mp4, .avi, .mov, .mkv)
3. Click "Start Processing"
4. View results and predictions

### Option 2: Using API Directly

```powershell
# Test with cURL
curl -X POST http://localhost:8000/api/predict -F "file=@test_video.mp4"
```

## 📊 View Results in MLflow

1. Go to your DagsHub repository
2. Click **"Experiments"** tab
3. View all inference runs with:
   - Parameters (model version, video path)
   - Metrics (inference time, frames processed)
   - Artifacts (results JSON, predictions)

## 🔧 Configuration Details

### Environment Variables

| Variable              | Description           | Example                                           |
| --------------------- | --------------------- | ------------------------------------------------- |
| `DAGSHUB_USERNAME`    | Your DagsHub username | `aadilsal234`                                     |
| `DAGSHUB_PAT`         | Personal Access Token | Get from https://dagshub.com/user/settings/tokens |
| `MLFLOW_TRACKING_URI` | MLflow server URL     | `https://dagshub.com/{user}/{repo}.mlflow`        |
| `MLFLOW_MODEL_NAME`   | Model registry name   | `tf-object-detection`                             |
| `MODEL_VERSION`       | Model stage/version   | `Staging`, `Production`, or `1`, `2`              |
| `DEVICE`              | Inference device      | `cpu` or `cuda`                                   |

### Video Processing Settings

- **Supported formats**: `.mp4`, `.avi`, `.mov`, `.mkv`
- **Max file size**: 500 MB (configurable via `MAX_VIDEO_SIZE`)
- **Frame sampling**: Every frame (configurable in `video_processor.py`)
- **Preprocessing**: Resize to 640×360, normalize to [0,1]

### Model Information

- **Architecture**: TensorFlow EfficientDet Lite2
- **Source**: TensorFlow Hub
- **Input size**: 320×320
- **Task**: Object detection
- **Output**: Bounding boxes, class labels, confidence scores

## 🛠️ Troubleshooting

### MLflow Connection Error (404)

**Problem**: `API request failed with error code 404`

**Solutions**:

1. Verify repository exists on DagsHub
2. Check MLflow is enabled in repo settings
3. Confirm `.env` has correct repository name
4. See `DAGSHUB_SETUP_GUIDE.md` for detailed steps

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'mlflow'`

**Solution**:

```powershell
pip install -r mlops/requirements.txt
```

### Model Loading Fails

**Problem**: `Model not found in registry`

**Solution**:

1. Run `python mlops/train_and_register_model.py` first
2. Check model appears in MLflow UI
3. Verify `MLFLOW_MODEL_NAME` matches registry name

### Backend Server Won't Start

**Problem**: Port already in use

**Solution**:

```powershell
# Use different port
$env:PORT="8001"
python mlops/server.py
```

Then update `.env`:

```env
BACKEND_URL=http://localhost:8001
```

## 📁 Data Storage

### Upload Directory

- Location: `mlops/data/uploads/`
- Purpose: Temporary storage for uploaded videos
- Cleanup: Manual (consider periodic cleanup script)

### Results Directory

- Location: `mlops/data/results/`
- Purpose: Inference results in JSON format
- Format: `results_{run_id}.json`

## 🔐 Security Notes

- **Never commit `.env`** to version control
- Add `.env` to `.gitignore`
- Rotate PAT tokens regularly
- Use environment variables in production
- Validate all file uploads (size, format, content)

## 📦 Dependencies

### Backend (`mlops/requirements.txt`)

```
mlflow>=2.8.0
dagshub>=0.2.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
python-dotenv>=1.0.0
fastapi>=0.95.0
uvicorn>=0.22.0
tensorflow>=2.13.0
tensorflow-hub>=0.14.0
python-multipart>=0.0.6
```

### Frontend (`frontend/requirements.txt`)

```
streamlit>=1.28.0
requests>=2.31.0
python-dotenv>=1.0.0
```

## 🚢 Deployment

### Docker (Coming Soon)

```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY mlops/ ./mlops/
COPY .env .env
RUN pip install -r mlops/requirements.txt
CMD ["python", "mlops/server.py"]
```

### Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets from `.env`
4. Deploy

## 📝 Next Steps

- [ ] Add batch processing for multiple videos
- [ ] Implement async job queue (Celery/RQ)
- [ ] Add video annotation overlay
- [ ] Export annotated video results
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Model performance monitoring
- [ ] A/B testing framework

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📄 License

[Add your license here]

## 📞 Support

- **DagsHub Docs**: https://dagshub.com/docs
- **Discord**: https://discord.com/invite/9gU36Y6
- **Issues**: Create an issue in this repository

---

**Built with** ❤️ using MLflow, DagsHub, TensorFlow, FastAPI, and Streamlit
