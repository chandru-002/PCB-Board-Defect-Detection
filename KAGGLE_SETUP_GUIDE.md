# 🔽 Kaggle Dataset Auto-Download Setup Guide
## PCB Defect Detection Project

---

## ✨ What's New?

You can now **automatically download the Kaggle dataset** without manual downloading!

The project now includes:
- ✅ `download_dataset.py` - Downloads from Kaggle automatically
- ✅ `setup.py` - One-click setup for everything

---

## 🚀 Fastest Setup (1 Command)

```bash
python setup.py
```

This automatically:
1. ✅ Installs all Python packages
2. ✅ Downloads dataset from Kaggle
3. ✅ Organizes dataset structure
4. ✅ Verifies everything works
5. ✅ Ready to train!

---

## 📋 Step-by-Step Setup

### Step 1: Get Kaggle API Credentials

**Why?** Kaggle API allows automatic downloads

**How:**
1. Go to: https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. File `kaggle.json` downloads to your computer
5. Move it to: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - macOS: `~/.kaggle/kaggle.json`
   - Linux: `~/.kaggle/kaggle.json`

**Verify it worked:**
```bash
cat ~/.kaggle/kaggle.json    # macOS/Linux
type %USERPROFILE%\.kaggle\kaggle.json    # Windows PowerShell
```

### Step 2: Run Automated Setup

```bash
# Install required Kaggle library
pip install kagglehub

# Run one-click setup
python setup.py
```

**What happens:**
```
════════════════════════════════════════════════════════════════
  🚀 PCB DEFECT DETECTION - AUTOMATED SETUP
════════════════════════════════════════════════════════════════

▶▶ STEP 1: Checking Python Installation
   Python Version: 3.10.5
   ✅ Python version compatible

▶▶ STEP 2: Installing Python Packages
   Installing packages: tensorflow, opencv, numpy, pandas...
   ✅ All dependencies installed

▶▶ STEP 3: Checking Kaggle API Credentials
   ✅ Kaggle API credentials found

▶▶ STEP 4: Downloading Dataset from Kaggle
   📥 Downloading PCB defects dataset from Kaggle...
   Dataset: akhatova/pcb-defects
   ⏳ Downloading... (this may take a few minutes)
   ✅ Dataset downloaded successfully

▶▶ STEP 5: Verifying Installation
   Checking Python packages...
   ✅ TensorFlow
   ✅ OpenCV
   ✅ NumPy
   ✅ Pandas
   ✅ Matplotlib
   ✅ All packages installed

   Checking dataset structure...
   ✅ missing_hole: 143 images
   ✅ mouse_bite: 152 images
   ✅ open_circuit: 112 images
   ✅ short: 97 images
   ✅ spur: 145 images
   ✅ spurious_copper: 119 images
   ✅ Dataset verified (768 total images)

🎉 SETUP COMPLETE!
════════════════════════════════════════════════════════════════
```

---

## Alternative: Manual Dataset Download

### Option A: Download Just the Dataset

```bash
python download_dataset.py
```

Requires: Kaggle API credentials set up first

### Option B: Download Manually from Kaggle Website

1. Go to: https://www.kaggle.com/datasets/akhatova/pcb-defects
2. Click "Download"
3. Extract to project folder as `dataset/`

Your folder should look like:
```
dataset/
├── missing_hole/        (143 images)
├── mouse_bite/          (152 images)
├── open_circuit/        (112 images)
├── short/               (97 images)
├── spur/                (145 images)
└── spurious_copper/     (119 images)
```

---

## 🎯 Three Setup Options

### Option 1: Fully Automated (Recommended)
```bash
python setup.py
```
✅ Everything automatic  
✅ Fastest  
❌ Requires Kaggle API

### Option 2: Just Download Dataset
```bash
pip install kagglehub
python download_dataset.py
```
✅ Fast  
❌ Requires manual pip install

### Option 3: Manual Download
1. Download from Kaggle website
2. Extract to `dataset/` folder
3. Run `python train_model.py`

✅ No dependencies  
❌ Manual steps

---

## ✅ Verify Installation

After setup, verify by running:

```bash
# Check dataset
python explore_dataset.py

# Train model
python train_model.py

# Make predictions
python predict_pcb.py
```

---

## 🚨 Troubleshooting

### Problem: "ImportError: No module named 'kagglehub'"

**Solution:**
```bash
pip install kagglehub
```

### Problem: "Kaggle API error - credentials not found"

**Solution:**
1. Download API token from: https://www.kaggle.com/account
2. Move to correct folder:
   ```bash
   # Windows PowerShell
   Move-Item C:\Users\<username>\Downloads\kaggle.json -Destination $env:USERPROFILE\.kaggle\
   
   # macOS/Linux
   mv ~/Downloads/kaggle.json ~/.kaggle/
   ```

### Problem: "Dataset folder already exists"

**Solution:**
That's fine! Script will skip download and use existing dataset.

### Problem: Network timeout during download

**Solution:**
- Try again (network may be slow)
- Or download manually from website
- Download is ~500 MB, may take 5-15 min

---

## 📊 Dataset Info

**Dataset:** PCB Defects from Kaggle  
**Source:** https://www.kaggle.com/datasets/akhatova/pcb-defects  
**Size:** ~500 MB  
**Images:** ~770 images  
**Classes:** 6 defect types  
**Format:** JPEG images

**Defect Classes:**
- missing_hole - PCB holes that are missing
- mouse_bite - Material bites along traces
- open_circuit - Breaks in traces
- short - Unintended connections
- spur - Unwanted protruding traces
- spurious_copper - Unexpected copper deposits

---

## 🎓 How It Works

### `download_dataset.py`
```
1. Checks kagglehub installation
2. ✅ Installs if needed
3. Verifies Kaggle credentials
4. Downloads dataset: akhatova/pcb-defects
5. Organizes into class folders
6. Ready for training!
```

### `setup.py`
```
1. Checks Python version
2. Installs dependencies (pip install -r requirements.txt)
3. Checks Kaggle credentials
4. Downloads dataset (if credentials found)
5. Verifies all packages
6. Verifies dataset structure
7. Shows next steps
```

---

## ⚡ Quick Commands Reference

```bash
# One-click setup (recommended)
python setup.py

# Just download dataset
python download_dataset.py

# Explore dataset
python explore_dataset.py

# Train model
python train_model.py

# Make predictions
python predict_pcb.py --folder dataset/test/ --save-csv
```

---

## 📚 What's Different from Previous Setup?

| Old Method | New Method |
|-----------|-----------|
| Download manually from website | Automatic via kagglehub |
| Manual folder organization | Automatic organization |
| Multiple steps | One command: `python setup.py` |
| Error-prone | Validated & verified |

---

## 💡 ProTips

**Tip 1: Cache Download**
```bash
# First run: downloads ~500 MB
python setup.py

# Second run: uses cached version (instant)
python setup.py
```

**Tip 2: Redownload if Corrupted**
```bash
rm -r dataset/              # Remove old dataset
python download_dataset.py   # Download fresh copy
```

**Tip 3: Use in CI/CD**
```bash
# Automated setup in GitHub Actions, Jenkins, etc.
python setup.py
python train_model.py
```

---

## 🔐 Security Note

Your `kaggle.json` contains your credentials. Keep it safe:
- ✅ Never commit to Git
- ✅ Never share publicly
- ✅ Store securely locally

---

## 📞 Support

**If setup fails:**
1. Check error message carefully
2. Try step-by-step: `pip install -r requirements.txt` first
3. Check Kaggle credentials are in correct location
4. Try manual download from website as backup

**Dataset issues:**
- Go to: https://www.kaggle.com/datasets/akhatova/pcb-defects
- Report any issues there

---

## ✨ Next: Train Your Model!

After setup completes:

```bash
# Explore dataset (optional)
python explore_dataset.py

# Train the model
python train_model.py

# This will:
# ✅ Load 768 images
# ✅ Build CNN model
# ✅ Train for ~20 epochs
# ✅ Save trained model to models/
# ✅ Create accuracy/loss graphs
```

**Training time:**
- GPU: 5-15 minutes ⚡
- CPU: 30-60 minutes ⏳

---

## 📋 Checklist

Before training, verify:

- [ ] Python 3.8+ installed
- [ ] Kaggle API credentials set up (`~/.kaggle/kaggle.json`)
- [ ] Run: `python setup.py` (completes successfully)
- [ ] Dataset folder exists with 6 subfolders
- [ ] All packages installed (numpy, tensorflow, opencv, etc.)

---

**You're ready to train!** 🚀

Next: `python train_model.py`
