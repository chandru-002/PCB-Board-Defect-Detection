# PCB Board Defect Detection - Production Setup

## Project Cleaned for Netlify Deployment ✅

### Removed Files (Development Only):
- `download_dataset.py` - Dataset download script
- `explore_dataset.py` - Data exploration script  
- `QUICKSTART.py` - Quick start guide
- `train_model.py` - Model training script
- `setup.py` - Development setup
- Summary and guide documentation files
- `__pycache__/` - Python cache
- `output/` - Temporary output files
- `pyrightconfig.json` - Development config

### Optimized Dependencies:
- Removed: matplotlib, pandas, seaborn, scikit-learn (dev-only)
- Kept: Flask, TensorFlow, Pillow, numpy (production essentials)

### New Files Created:
- `netlify.toml` - Netlify configuration
- `runtime.txt` - Python version specification
- `Procfile` - Server process configuration
- `.gitignore` - Updated to exclude large files

### Deployment Ready:
- ✅ Cleaned project structure
- ✅ Optimized dependencies
- ✅ Netlify configuration added
- ✅ Large data excluded from repository

## Deployment Steps:

1. **Connect to Netlify:**
   - Go to https://app.netlify.com
   - Click "Add new site" → "Import an existing project"
   - Select your GitHub repository
   - Build command: `pip install -r requirements.txt`
   - Publish directory: `.`

2. **Environment Variables (Set in Netlify Dashboard):**
   - `ENABLE_MODEL_INFERENCE=0` (uses reference matcher for faster startup)
   - `APP_HOST=0.0.0.0`
   - `APP_PORT=8888`

3. **Deploy:**
   - Push changes to GitHub
   - Netlify will automatically build and deploy

## Notes:
- Model inference disabled by default for fast startup
- Uses lightweight reference-based matching algorithm
- Supports image upload and PCB defect classification
- Reference dataset included in `PCB_DATASET/images/`
