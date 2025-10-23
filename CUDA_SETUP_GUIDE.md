# CUDA Setup Guide for GTX 1650 Ti

## Current Situation

❌ **Problem**: TensorFlow 2.20.0 for Python 3.13 on Windows doesn't have GPU support  
✅ **Your GPU**: NVIDIA GeForce GTX 1650 Ti (Compute 7.5, CUDA 13.0)  
✅ **Driver**: 580.88 (excellent, up-to-date)  

## The Issue

TensorFlow stopped providing GPU-enabled Windows wheels for newer Python versions (3.11+). The current TensorFlow 2.20.0 is CPU-only on Windows with Python 3.13.

## Solutions (Ranked by Practicality)

---

### **Option 1: Google Colab (RECOMMENDED)** ⭐⭐⭐⭐⭐

**Why**: Free T4 GPU (faster than GTX 1650 Ti!), no installation, ready in 5 minutes

**Steps**:
1. I create a Colab notebook for you
2. You upload it to Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Upload data or connect Google Drive
5. Run training (5-7 hours total)
6. Download trained model

**Pros**:
- FREE T4 GPU (16GB VRAM)
- Faster than your local GPU
- No installation
- Ready to use

**Cons**:
- 12-hour session limit
- Need to upload ~100MB data

---

### **Option 2: Use Python 3.10 + TensorFlow 2.15** ⭐⭐⭐⭐

**Why**: Last TensorFlow version with official GPU support on Windows

**Steps**:

1. **Create new Python 3.10 environment**:
   ```powershell
   # Download Python 3.10 from python.org
   # Install to C:\Python310
   
   # Create new virtual environment
   C:\Python310\python.exe -m venv .venv-gpu
   ```

2. **Activate and install**:
   ```powershell
   .\.venv-gpu\Scripts\activate
   pip install tensorflow==2.15.0 pandas numpy scikit-learn keras
   ```

3. **Verify GPU**:
   ```powershell
   python check_gpu.py
   ```

4. **Train**:
   ```powershell
   python src/train_universal_lstm_optimized.py
   ```

**Time**: 30 minutes setup, then 5-10 hours training

---

### **Option 3: Windows Subsystem for Linux (WSL2)** ⭐⭐⭐

**Why**: Use Linux TensorFlow (better GPU support)

**Steps**:

1. **Install WSL2**:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

2. **Install CUDA in WSL**:
   ```bash
   # In Ubuntu terminal
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-6
   ```

3. **Install TensorFlow**:
   ```bash
   pip install tensorflow[and-cuda]
   ```

4. **Copy project and train**:
   ```bash
   python train_universal_lstm_optimized.py
   ```

**Time**: 1 hour setup, then 5-10 hours training

---

### **Option 4: Docker with GPU** ⭐⭐

**Why**: Isolated environment with GPU support

**Steps**:

1. Install Docker Desktop
2. Enable WSL2 backend
3. Install NVIDIA Container Toolkit
4. Use TensorFlow GPU Docker image

**Time**: 1-2 hours setup

---

### **Option 5: Keep Current Setup (CPU Training)** ⭐⭐⭐

**Why**: No changes needed, will complete eventually

**Current Status**:
- Training on CPU
- ~30-45 min per epoch
- ~75-113 hours total (3-5 days)

**Optimization**:
- Train overnight for a few days
- OR reduce to top 50-100 stocks (6-16 hours)

---

## My Strong Recommendation

### **Use Google Colab (Free)** 🚀

**Why it's the best choice**:
1. ✅ Faster than your GTX 1650 Ti (T4 has 16GB vs your 4GB)
2. ✅ FREE - no cost
3. ✅ Ready in 5 minutes
4. ✅ No local installation/configuration
5. ✅ Complete training in one session (7-8 hours)
6. ✅ Can monitor from phone/anywhere

**I can create the Colab notebook for you right now!**

---

## Comparison Table

| Method | GPU Speed | Setup Time | Cost | Difficulty | Total Time |
|--------|-----------|------------|------|------------|------------|
| **Colab Free** | T4 (fast) | 5 min | FREE | ⭐ Easy | **5-8 hours** |
| Python 3.10 | GTX 1650 Ti | 30 min | FREE | ⭐⭐ Medium | 5.5-10.5 hours |
| WSL2 | GTX 1650 Ti | 60 min | FREE | ⭐⭐⭐ Hard | 6-11 hours |
| Docker | GTX 1650 Ti | 120 min | FREE | ⭐⭐⭐⭐ Very Hard | 7-12 hours |
| **CPU (current)** | None | 0 min | FREE | ⭐ Easy | **75-113 hours** |
| Top 50 CPU | None | 2 min | FREE | ⭐ Easy | 6-8 hours |

---

## What Should You Do?

### Recommended Path:

**For This Training Session**: Use Google Colab (I'll create notebook)
- Complete in 5-8 hours
- No local changes needed
- Fastest result

**For Future Projects**: Install Python 3.10 environment
- One-time 30-min setup
- Always have GPU ready
- Use local hardware

---

## Next Steps - Choose One:

### A. Google Colab (5 min to start) ⭐ **BEST**
Say: **"Create Colab notebook"**
- I'll create a ready-to-run notebook
- You upload to Colab
- Train in 5-8 hours

### B. Python 3.10 Setup (30 min)
Say: **"Help me install Python 3.10"**
- Install Python 3.10
- Create GPU environment
- Train locally with GPU

### C. Continue CPU Training
Say: **"Continue CPU training"**
- Current training completes in 3-5 days
- No changes needed

### D. Train on Fewer Stocks
Say: **"Train top 50 stocks"**
- Modify script
- Complete in 6-8 hours on CPU
- Good for testing

---

**What would you like to do?**
