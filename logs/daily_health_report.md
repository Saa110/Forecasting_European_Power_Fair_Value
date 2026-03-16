# Daily Pipeline Health Report
**Generated:** 2026-03-15T18:02:37.997928
**Mode:** Automated (rule-based fallback — set OPENAI_API_KEY for LLM reports)

## 1. Data Quality Summary
- ✅ No duplicate timestamps detected
- [DST/Anomaly] prices: No duplicate timestamps ✓
- [DST/Anomaly] prices: WARNING gap of 1.0h at 2023-12-31 23:00:00+00:00 (possible DST spring-forward or data outage)
- [DST/Anomaly] prices: WARNING gap of 1.0h at 2024-01-01 00:00:00+00:00 (possible DST spring-forward or data outage)

## 2. Model Performance
- **Naive Persistence**: sMAPE=28.58%, MAE=22.72 €/MWh
- **Ridge (ARX)**: sMAPE=26.44%, MAE=16.99 €/MWh
- **LightGBM**: sMAPE=17.86%, MAE=11.97 €/MWh
- **XGBoost**: sMAPE=17.52%, MAE=11.77 €/MWh
- **Ensemble**: sMAPE=17.71%, MAE=11.77 €/MWh
- ✅ Zero quantile crossings — probabilistic outputs well-calibrated

## 3. Trading Signal Summary
- **2026-W09**: Mild Buy (RP=-3.5 €/MWh, Conviction: Half position)
- **2026-W10**: Strong Sell (Overvalued) (RP=+13.4 €/MWh, Conviction: Full position)
- **2026-W11**: Mild Buy (RP=-2.1 €/MWh, Conviction: Half position)

## 4. Action Items
- 📋 REMIT parser using rule-based fallback — set OPENAI_API_KEY for LLM classification