"""
PCB DEFECT DETECTION - QUICK START GUIDE
=========================================
Follow this step-by-step for fastest setup!

Author: AI-ML Project
Date: 2024
"""

import sys
import os
print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║        🚀 PCB DEFECT DETECTION - QUICK START GUIDE 🚀         ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

This is the fastest way to get your project running!

STEP 1: INSTALL PYTHON
══════════════════════════════════════════════════════════════════

   Go to: https://www.python.org/downloads/
   
   ✅ Download Python 3.9 or higher
   ✅ During installation, CHECK "Add Python to PATH"
   ✅ CHECK "Install pip"
   
   Verify:
   $ python --version
   $ pip --version


STEP 2: DOWNLOAD DATASET
══════════════════════════════════════════════════════════════════

   Option A (Easy - Recommended):
   • Go to: https://www.kaggle.com/
   • Search: "PCB Defect Detection"
   • Download dataset (choose "Tangling" or similar)
   • Extract to your project folder
   
   Result should look like:
   
   PCB_Project/
   ├── dataset/
   │   ├── missing_hole/      (contains images)
   │   ├── mouse_bite/
   │   ├── open_circuit/
   │   ├── short/
   │   ├── spur/
   │   └── spurious_copper/


STEP 3: SETUP PROJECT FOLDER
══════════════════════════════════════════════════════════════════

   $ mkdir PCB_Project
   $ cd PCB_Project
   $ code .              # Opens in VS Code


STEP 4: INSTALL LIBRARIES
══════════════════════════════════════════════════════════════════

   In VS Code Terminal (Ctrl + `):
   
   $ pip install -r requirements.txt
   
   This installs:
   ✅ TensorFlow (deep learning)
   ✅ OpenCV (image processing)
   ✅ NumPy (math)
   ✅ Matplotlib (graphs)
   ... and more!


STEP 5: TRAIN MODEL
══════════════════════════════════════════════════════════════════

   In VS Code Terminal:
   
   $ python train_model.py
   
   What happens:
   ✅ Loads dataset (~900 images)
   ✅ Splits into training/validation/test
   ✅ Trains CNN (takes 5-60 min depending on PC)
   ✅ Saves model to models/pcb_defect_model.h5
   ✅ Saves graphs to output/


STEP 6: TEST MODEL
══════════════════════════════════════════════════════════════════

   Option A - Single image:
   
   $ python predict_pcb.py --image dataset/missing_hole/image001.jpg --visualize
   
   
   Option B - Batch test:
   
   $ python predict_pcb.py --folder dataset/test/ --save-csv
   
   
   Option C - Default test:
   
   $ python predict_pcb.py


THAT'S IT! 🎉
══════════════════════════════════════════════════════════════════

Your model is trained and making predictions!

What you now have:
✅ models/pcb_defect_model.h5    (Your trained AI model)
✅ output/training_history.png   (Accuracy/Loss graphs)
✅ output/predictions.csv         (Batch predictions)


TROUBLESHOOTING
══════════════════════════════════════════════════════════════════

Problem: "ModuleNotFoundError: No module named 'tensorflow'"
Solution:
   $ pip install tensorflow

Problem: "Dataset folder not found"
Solution:
   Make sure your folder structure is correct:
   $ ls dataset/
   should show: missing_hole, mouse_bite, open_circuit, ...

Problem: "Memory error"
Solution:
   Edit train_model.py line:
   BATCH_SIZE = 16    (change from 32)

Problem: "Training is very slow"
Solution:
   This is normal on CPU. Use GPU if available.
   Or reduce dataset size for testing.


NEXT STEPS
══════════════════════════════════════════════════════════════════

1. Read the full README.md for detailed information
2. Check output/ folder for results and graphs
3. Try different prediction options
4. Modify hyperparameters to improve accuracy
5. Deploy as web app (optional)


✅ YOU'RE READY TO GO! 🚀

Questions? Check README.md for full documentation.
""")

# Auto-check if dataset exists

print("\n\n🔍 CHECKING YOUR SETUP:")
print("="*60)

# Check Python version
print(f"✅ Python {sys.version.split()[0]} detected")

# Check libraries
try:
    import tensorflow
    print(f"✅ TensorFlow {tensorflow.__version__} installed")
except:
    print("❌ TensorFlow not installed - run: pip install tensorflow")

try:
    import cv2
    print(f"✅ OpenCV installed")
except:
    print("❌ OpenCV not installed - run: pip install opencv-python")

# Check dataset
if os.path.exists("dataset"):
    folders = os.listdir("dataset")
    print(f"✅ Dataset folder found with {len(folders)} classes")
    for f in sorted(folders):
        if os.path.isdir(f"dataset/{f}"):
            images = len([i for i in os.listdir(
                f"dataset/{f}") if i.endswith(('.jpg', '.png'))])
            print(f"   • {f}: {images} images")
else:
    print("⚠️  Dataset folder not found - download from Kaggle")

# Check model
if os.path.exists("models/pcb_defect_model.h5"):
    print("✅ Trained model found - ready to make predictions")
else:
    print("⚠️  No trained model - run: python train_model.py")

print("\n" + "="*60)
print("✅ SETUP CHECK COMPLETE!")
print("="*60)
