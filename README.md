# XTB Transition State Screener
Screening 'likely-to-converge' proposed transition states as partially optimized by semiemperical quantum mechanics.

Uses Julia, Lux, and ADAM.

## Iteration 1 Notes - Machine Learning Subgroup Meeting
Could also try XGBoost or Random Forest.

Need to make goal of the model to be converge or not converge, NOT that it is valid. XTBTSScreener makes to judgement on validity, just if we will be wasting compute hours or not. This ties in better to performance engineering.

Need to calrify that there are different failure modes, SCF can fail (one step in the DFT can fail) or the calculation itself can fail (some meta reason).

Need to better define 'valid'.

Here is a reference paper that did something similar: https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00081

There would be a separate model that predicts if an xTB proposed transition state will produce a valid transition state.