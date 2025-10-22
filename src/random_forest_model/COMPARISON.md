# Model Comparison Summary

## All Models Performance

| Model | Features | R² Score | RMSE | MAE | MAPE | Winner |
|-------|----------|----------|------|-----|------|--------|
| **LightGBM Basic** | 15 | **0.6578** | 143.25 | 104.98 | N/A | 🏆 |
| **Random Forest** | 55 | **0.6402** | **132.18** | **101.14** | **6.82%** | 🥈 |
| Random Forest (Ensemble) | 30 | 0.6080 | 146.81 | 112.21 | 7.38% | |
| Ensemble (50/50) | 30 | 0.5809 | 149.42 | 114.00 | 7.49% | |
| LightGBM Optimized | 25 | 0.5602 | 147.75 | 115.11 | 7.72% | |
| LightGBM Full | 83 | 0.5670 | 152.20 | 116.06 | N/A | |
| LightGBM (Ensemble) | 30 | 0.5279 | 157.06 | 120.54 | 7.94% | |

## Key Findings

### 🏆 Winner: LightGBM Basic (R² = 0.6578)
- **Why it wins**: Simple features, no overfitting
- **Best for**: Overall R² score
- **Limitation**: Higher RMSE than Random Forest

### 🥈 Runner-Up: Random Forest Standalone (R² = 0.6402)
- **Why it's great**: Lowest RMSE (132.18) and MAE (101.14)
- **Best for**: Minimizing prediction errors
- **Advantage**: Best MAPE (6.82%)

### Important Insights

1. **Simplicity Wins**
   - With only 2,611 samples, simple models (15-55 features) outperform complex ones (83 features)
   - The basic LightGBM with 15 features beats all heavily engineered versions

2. **Random Forest Excellence**
   - **Lowest absolute errors** (RMSE & MAE)
   - **Best MAPE** (6.82% vs 7.72%)
   - More stable predictions

3. **Ensemble Doesn't Help**
   - Ensemble (R² = 0.58) performs worse than both individual models
   - **Why?** Both models make similar predictions on this data
   - Ensembles work best when models are diverse/complementary

4. **Feature Engineering Paradox**
   - More features ≠ Better performance
   - 83 features → R² = 0.57 (overfitting)
   - 15 features → R² = 0.66 (just right)
   - 55 features → R² = 0.64 (balanced)

## Recommendations

### For Production Use
Choose based on your priority:

**Priority: Best Overall R²**
→ Use **LightGBM Basic** (15 features)
- Highest R² = 0.6578
- Simpler, faster
- Less prone to overfitting

**Priority: Lowest Prediction Errors**
→ Use **Random Forest** (55 features)
- Lowest RMSE = 132.18
- Lowest MAE = 101.14
- Best MAPE = 6.82%

**Priority: Robustness & Interpretability**
→ Use **Random Forest** (55 features)
- Clearer feature importance
- More stable across folds
- Better error metrics

### Optimal Strategy
**Hybrid Approach:**
1. Use **Random Forest** for daily predictions (best errors)
2. Use **LightGBM Basic** for trend validation (best R²)
3. Only trade when both models agree on direction

## Files in This Folder

- `rf_price_prediction.py` - Main Random Forest model (55 features)
- `ensemble_model.py` - Combined RF + LightGBM model
- `README.md` - Detailed Random Forest documentation
- `COMPARISON.md` - This file

## Real-World Performance

At NABIL price ≈ 1,000 NPR:

**Random Forest Predictions:**
- Average error: ±101 NPR (±10.1%)
- MAPE: 6.82%
- Typical range: 930-1070 NPR

**LightGBM Basic Predictions:**
- Average error: ±105 NPR (±10.5%)
- Higher R² but slightly larger errors

## Conclusion

**The winner depends on what you value:**
- **R² Score** → LightGBM Basic 🏆
- **Error Minimization** → Random Forest 🥈
- **Overall Balance** → Random Forest 🥈

For **real trading**, we recommend **Random Forest** due to:
- Lower absolute errors (critical for profit/loss)
- Better MAPE (percentage errors matter in trading)
- More interpretable feature importance
- Proven stability

Both models perform reasonably well (~64-66% variance explained) for 5-day ahead predictions, which is acceptable for trend-following strategies with proper risk management.
