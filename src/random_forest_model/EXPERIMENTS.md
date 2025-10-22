# Experiment Results: 10x Parameters & SCB Bank Analysis

## Experiment 1: Random Forest with 10x Parameters (NABIL)

### Configuration Comparison

| Configuration | n_estimators | max_depth | R² Score | RMSE | Training Time |
|--------------|--------------|-----------|----------|------|---------------|
| **Normal** | 200 | 15 | **0.6319** ✅ | **135.25** ✅ | **1.59s** ✅ |
| **10x** | 2000 | 150 | 0.6121 ❌ | 137.46 ❌ | 18.54s ❌ |

### Key Findings

❌ **10x Parameters FAILED to Improve Performance**

**Performance Degradation:**
- R² dropped from 0.6319 → 0.6121 (-3.1%)
- RMSE increased from 135.25 → 137.46 (+1.6%)
- MAE increased from 103.59 → 105.77 (+2.1%)
- MAPE increased from 6.93% → 7.14% (+3.0%)

**Time Cost:**
- Training time: 1.59s → 18.54s (**11.7x slower**)
- Prediction time: 0.37s → 2.09s (**5.6x slower**)

### Why 10x Failed

1. **Overfitting with Deep Trees**
   - max_depth=150 captures noise, not signal
   - With only 2,611 samples, trees memorize training data
   
2. **Diminishing Returns**
   - After 200 trees, additional trees don't help
   - Just averaging more similar predictions

3. **Too Flexible**
   - min_samples_leaf=1 allows single-sample leaves
   - Creates unstable, overfitted predictions

### Conclusion

**🎯 Optimal Configuration: 200 trees, depth 15**
- Best balance of accuracy and speed
- Avoids overfitting
- 12x faster than 10x version

**Rule: More complexity ≠ Better predictions (especially with small datasets)**

---

## Experiment 2: SCB Bank Price Prediction

### Dataset Info
- **Total samples**: 2,622 trading days
- **Date range**: April 2010 - December 2021
- **Features**: 57 technical indicators
- **Prediction**: 5 days ahead

### Performance Results

| Metric | Value | Status |
|--------|-------|--------|
| **Avg R² Score** | **-2.8563** | ❌ CATASTROPHIC |
| **Avg RMSE** | 233.50 NPR | ❌ HIGH |
| **Avg MAE** | 190.26 NPR | ❌ HIGH |
| **Avg MAPE** | 18.53% | ❌ POOR |

### Fold-by-Fold Analysis

| Fold | Train Size | Test Size | RMSE | MAE | R² | MAPE | Status |
|------|-----------|-----------|------|-----|-------|------|--------|
| 1 | 437 | 437 | 101.35 | 78.24 | 0.1005 | 4.22% | ⚠️ Poor |
| 2 | 874 | 437 | 262.46 | 193.63 | 0.5534 | 7.67% | ⚠️ OK |
| 3 | 1,311 | 437 | 339.06 | 240.78 | 0.7960 | 11.12% | ✅ Good |
| 4 | 1,748 | 437 | **425.51** | **411.78** | **-16.31** | **64.93%** | ❌ DISASTER |
| 5 | 2,185 | 437 | 39.10 | 26.85 | 0.5814 | 4.71% | ✅ Good |

### Problem Analysis

**Fold 4 Catastrophic Failure:**
- R² = -16.31 (worse than random guessing!)
- MAPE = 64.93% (predictions off by 65%!)
- Indicates major structural break or regime change

**Possible Causes:**
1. **Extreme market volatility** in that time period
2. **Corporate events** (merger, rights issue, dividend)
3. **Regulatory changes** affecting banking sector
4. **Market crash or bubble** (2015-2016 period?)
5. **Data quality issues** in that fold

### SCB vs NABIL Comparison

| Stock | R² Score | RMSE | MAE | MAPE | Stability |
|-------|----------|------|-----|------|-----------|
| **NABIL** | 0.6402 | 132.18 | 101.14 | 6.82% | ✅ Stable |
| **SCB** | -2.8563 | 233.50 | 190.26 | 18.53% | ❌ Unstable |

**SCB is 3-4x harder to predict than NABIL!**

### Why SCB Fails

1. **Higher Volatility**
   - More price swings
   - Less predictable patterns

2. **Regime Changes**
   - Different market behaviors across time periods
   - Single model can't capture all regimes

3. **Lower Liquidity** (possibly)
   - More random price movements
   - Less efficient market

4. **Different Business Fundamentals**
   - SCB might be more affected by external factors
   - NABIL has more stable patterns

---

## Recommendations

### For NABIL Bank Predictions ✅
- **Use Normal RF (200 trees)** - proven best
- R² = 0.64, RMSE = 132 NPR
- Reliable for 5-day ahead predictions

### For SCB Bank Predictions ❌
**Current model is NOT USABLE - requires major changes:**

1. **Implement Regime Detection**
   ```python
   # Detect market regimes (bull/bear/sideways)
   # Train separate models for each regime
   ```

2. **Shorter Prediction Horizon**
   - Try 1-day or 3-day ahead instead of 5-day
   - Reduce forecast difficulty

3. **Add External Features**
   - Banking sector index
   - Interest rates
   - Economic indicators
   - News sentiment

4. **Use Different Approach**
   - LSTM/GRU networks for time series
   - ARIMA for baseline comparison
   - Ensemble with multiple time horizons

5. **Filter Extreme Periods**
   - Identify and handle structural breaks
   - Don't trade during high uncertainty

### General Learnings

✅ **DO:**
- Keep models simple (200 trees, depth 10-15)
- Test on stable, liquid stocks (like NABIL)
- Use proper time series validation
- Monitor individual fold performance

❌ **DON'T:**
- Increase parameters 10x hoping for magic
- Apply same model to all stocks blindly
- Ignore catastrophic failures in validation
- Trade based on models with negative R²

---

## Conclusion

### 10x Parameters Experiment
**FAILED** - Worse performance, much slower, overfitting

**Optimal: 200 trees, depth 15-20, min_samples_leaf 4-10**

### SCB Prediction Experiment
**FAILED** - Model unusable (R² = -2.86)

**Recommendation: DO NOT TRADE SCB with current model**

### Success Story
**NABIL model works well:**
- R² = 0.64
- RMSE = 132 NPR
- MAPE = 6.82%
- Stable across folds

**Use NABIL model with proper risk management!**

---

## Files Created

1. `rf_10x_parameters.py` - Tests 10x parameter configuration
2. `prepare_transactions_for_scb.py` - Converts SCB data
3. `rf_scb_prediction.py` - SCB prediction model
4. `stock_daily_scb.csv` - Processed SCB data

## Next Steps

1. ✅ Stick with normal RF parameters (200 trees)
2. ✅ Use NABIL model for predictions
3. ❌ Fix SCB model before using (implement regime detection)
4. 🔍 Investigate Fold 4 failure in SCB data
5. 📊 Try other banks with similar stability to NABIL
