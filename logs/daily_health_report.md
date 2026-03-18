# Daily Pipeline Health Report
**Generated:** 2026-03-18T14:08:15.676208
**Mode:** Automated (rule-based fallback — set OPENAI_API_KEY for LLM reports)

## 1. Data Quality Summary
- ✅ No duplicate timestamps detected
- ✅ No unexpected gaps in time series
- [DST/Anomaly] prices: No duplicate timestamps ✓
- [DST/Anomaly] prices: No unexpected gaps ✓
- [DST/Anomaly] prices: Detected resolutions → [12, 24, 48, 96] records/day

## 2. Model Performance
- **Naive Persistence**: sMAPE=30.92%, MAE=24.23 €/MWh
- **Ridge (ARX)**: sMAPE=24.85%, MAE=16.64 €/MWh
- **LightGBM**: sMAPE=16.84%, MAE=11.36 €/MWh
- **XGBoost**: sMAPE=16.39%, MAE=10.88 €/MWh
- **Ensemble**: sMAPE=16.4%, MAE=11.02 €/MWh
- ✅ Zero quantile crossings — probabilistic outputs well-calibrated

## 3. Trading Signal Summary
- **2026-W10**: Strong Sell (Overvalued) (RP=+12.5 €/MWh, Conviction: Full position)
- **2026-W11**: Mild Sell (RP=+0.4 €/MWh, Conviction: Half position)
  - ⚠️ Risk premium < 2 €/MWh — insufficient edge after costs
- **2026-W12**: Mild Buy (RP=-1.6 €/MWh, Conviction: Half position)
  - ⚠️ Risk premium < 2 €/MWh — insufficient edge after costs

## 4. Action Items
- 📋 REMIT parser using rule-based fallback — set OPENAI_API_KEY for LLM classification