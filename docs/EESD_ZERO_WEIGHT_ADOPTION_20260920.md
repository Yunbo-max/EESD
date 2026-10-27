# EESD Zero-Positive-Training-Weight Adoption

Status: ADOPTED.

When a scored correction population for an update rule contains no positive-weight
training rows, the equal-budget training outcome is non-estimable. The update is
recorded as a `eesd-zero-training-outcome-v1` receipt with `status`
`non_estimable_zero_positive_training_weight`, no adapter, and no efficacy claim.
No fallback population is substituted and no descriptive estimate is adopted.
