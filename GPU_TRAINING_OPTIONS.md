# GPU Training Options for Universal LSTM

## Current Situation

✅ **NVIDIA GPU Available**: GTX 1650 Ti (4GB VRAM, CUDA 13.0)  
❌ **TensorFlow GPU Support**: Not enabled (built without CUDA)  
🟡 **Current Status**: Training on CPU (slower, but working)

---

## Option 1: Install CUDA Toolkit (Local GPU) ⭐ **BEST for Long-term**

### Pros
- **10-15x faster** than CPU
- Local control, no upload needed
- No time limits
- Keep using after project

### Cons
- Requires ~6GB download (CUDA + cuDNN)
- One-time setup (~30 minutes)

### Installation Steps

1. **Download CUDA Toolkit 12.x**
   - Go to: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → 11 → exe (network)
   - Install (will take 15-20 minutes)

2. **Download cuDNN**
   - Go to: https://developer.nvidia.com/cudnn
   - Download cuDNN v9.x for CUDA 12.x
   - Extract and copy files to CUDA directory

3. **Reinstall TensorFlow**
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-gpu  # or tensorflow[and-cuda]
   ```

4. **Verify GPU**
   ```bash
   python check_gpu.py
   ```

### Expected Speed
- **CPU**: ~30-45 min per epoch = **75-113 hours total** (3-5 days!)
- **GPU**: ~2-4 min per epoch = **5-10 hours total**

---

## Option 2: Google Colab (Free GPU) ⭐ **BEST for Quick Results**

### Pros
- **FREE** T4 GPU (faster than your GTX 1650 Ti!)
- No installation needed
- 2-3 min per epoch = **5-7.5 hours total**
- Ready in 5 minutes

### Cons
- 12-hour session limit (need to resume if interrupted)
- Need to upload data (~100MB)
- Requires Google account

### How to Use

1. **Upload Colab Notebook** (I'll create this for you)
2. **Enable GPU**: Runtime → Change runtime type → GPU → Save
3. **Upload data** to Google Drive or Colab
4. **Run cells** - automatic training
5. **Download trained model**

**I can create a ready-to-use Colab notebook for you!**

---

## Option 3: Reduce Dataset Size (Quick Training on CPU)

### Pros
- No setup needed
- Trains in reasonable time on CPU
- Good for testing/iteration

### Cons
- Less comprehensive model
- May miss some stock patterns
- Lower accuracy

### Options

#### A. Train on Top 50 Stocks (~6-8 hours on CPU)
```python
# Modify train_universal_lstm_optimized.py
top_50 = pd.read_csv('data/processed/conversion_summary.csv')
top_50 = top_50.nlargest(50, 'days')['symbol'].tolist()
stock_symbols = top_50  # Use this instead of loading all
```

#### B. Train on Top 100 Stocks (~12-16 hours on CPU)
```python
top_100 = summary.nlargest(100, 'days')['symbol'].tolist()
```

#### C. Train on Banking Sector Only (~4-6 hours on CPU)
```python
banks = ['NABIL', 'SCB', 'EBL', 'PCBL', 'SRBL', 'NIB', 'SBI', 
         'NBB', 'HBL', 'KBL', 'LBL', 'CZBIL', 'ADBL', 'MEGA']
```

---

## Option 4: Cloud GPU (Paid but Cheap)

### Providers
- **Google Colab Pro**: $10/month, no time limits
- **AWS EC2 g4dn.xlarge**: ~$0.50/hour = $2.50 for 5 hours
- **Google Cloud GPU**: ~$0.45/hour
- **Kaggle**: Free P100 GPU, 30 hours/week

### Best for
- One-time training
- Need to train full dataset
- Don't want to install CUDA

---

## Comparison Table

| Option | Cost | Setup Time | Training Time | Total Time | Difficulty |
|--------|------|------------|---------------|------------|------------|
| **CPU (current)** | Free | 0 min | 75-113 hours | 75-113 hours | ⭐ Easy |
| **Local GPU** | Free | 30 min | 5-10 hours | 5.5-10.5 hours | ⭐⭐⭐ Hard |
| **Colab Free** | Free | 5 min | 5-7.5 hours | 5-8 hours | ⭐ Easy |
| **Colab Pro** | $10 | 5 min | 5-7.5 hours | 5-8 hours | ⭐ Easy |
| **AWS GPU** | ~$2.50 | 10 min | 5-8 hours | 5-8 hours | ⭐⭐ Medium |
| **Top 50 CPU** | Free | 2 min | 6-8 hours | 6-8 hours | ⭐ Easy |
| **Top 100 CPU** | Free | 2 min | 12-16 hours | 12-16 hours | ⭐ Easy |

---

## My Recommendation

### For This Project (287 stocks, full training)

**Best Choice: Google Colab (Free)**
- FREE T4 GPU
- Ready in 5 minutes
- Complete training in 5-7 hours
- No installation hassle
- I can create the notebook for you!

**Alternative: Local CUDA (if you'll do more ML projects)**
- One-time setup (30 min)
- Always have GPU available
- Good investment for future projects
- Slightly slower than Colab but no limits

### For Quick Iteration (testing/development)

**Best Choice: Top 50 Stocks on CPU**
- Modify script to use top 50 stocks
- Train overnight (6-8 hours)
- Good enough for testing
- No setup needed

---

## What to Do Now?

### Option A: Let CPU Training Continue ✅
Your current training is running. Let it continue overnight. Check progress with:
```bash
# In another terminal
Get-Process python | Select-Object CPU,WorkingSet,ProcessName
```

### Option B: Stop and Use Google Colab 🚀
1. Press `Ctrl+C` to stop current training
2. I'll create a Colab notebook for you
3. Upload to Colab and train (5-7 hours)
4. Download trained model

### Option C: Stop and Train on Smaller Dataset ⚡
1. Press `Ctrl+C` to stop
2. I'll modify script for top 50/100 stocks
3. Train overnight (6-16 hours)
4. Still get good results

### Option D: Install CUDA for Local GPU 💪
1. Keep current training running
2. Install CUDA toolkit (~30 min)
3. Reinstall TensorFlow
4. Train again with GPU (5-10 hours)

---

## Quick Decision Guide

**I want the fastest training possible** → Google Colab (FREE, ready in 5 min)  
**I want to use my GPU eventually** → Install CUDA (30 min setup, then always fast)  
**I'm okay with overnight training** → Let current CPU training continue  
**I want to test quickly** → Top 50 stocks on CPU (6-8 hours)  

---

## Files Created

- ✅ `check_gpu.py` - Test GPU detection
- ✅ `src/train_universal_lstm_optimized.py` - Auto-detects GPU/CPU
- ⏳ Google Colab notebook (can create if needed)
- ⏳ Top-50 stocks script (can create if needed)

---

**Which option would you like to pursue?**
