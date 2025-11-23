# DagsHub Repository Setup Guide

## Issue: Repository Not Found (404 Error)

The MLflow connection test failed because the repository doesn't exist on DagsHub yet.

## Solution: Create DagsHub Repository

### Step 1: Create New Repository on DagsHub

1. Go to: **https://dagshub.com/repo/create**
2. Fill in the repository details:
   - **Repository name**: `FairPlayReviewSystem` (or your preferred name)
   - **Description**: "Video processing ML pipeline with MLflow integration"
   - **Visibility**: Private or Public
   - **Initialize with**: README (optional)
3. Click **Create Repository**

### Step 2: Update .env File

After creating the repository, update your `.env` file with the correct repository name:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/aadilsal234/FairPlayReviewSystem.mlflow
```

**Important**: Replace `FairPlayReviewSystem` with whatever name you chose in Step 1.

### Step 3: Enable MLflow in DagsHub

1. Go to your new repository: `https://dagshub.com/aadilsal234/FairPlayReviewSystem`
2. Click on the **"Experiments"** or **"MLflow"** tab
3. If prompted, click **"Enable MLflow"** or **"Start Using MLflow"**

### Step 4: Re-run Connection Test

```powershell
python test_mlflow_connection.py
```

## Alternative: Use Existing Repository

If you already have a DagsHub repository, update `.env` to point to it:

```env
# Example if your repo is named MLOPS_Proj:
MLFLOW_TRACKING_URI=https://dagshub.com/aadilsal234/MLOPS_Proj.mlflow
```

## Verification Checklist

- [ ] DagsHub repository created
- [ ] MLflow enabled in repository settings
- [ ] `.env` file updated with correct repo name
- [ ] `test_mlflow_connection.py` runs successfully
- [ ] MLflow UI accessible at tracking URI

## Next Steps After Setup

Once the repository is created and connection test passes:

1. Run `python mlops/train_and_register_model.py` to register the TensorFlow model
2. Verify model appears in MLflow Model Registry
3. Test model loading with `python mlops/test_model_loading.py`
4. Start FastAPI server and test inference pipeline

## Need Help?

- DagsHub Docs: https://dagshub.com/docs
- Discord Support: https://discord.com/invite/9gU36Y6
- Email: support@dagshub.com
