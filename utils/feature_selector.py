"""Re-export shim — feature selection has moved to feature_processing.feature_selector."""
from feature_processing.feature_selector import (  # noqa: F401
    TopKFeatureSelector,
    TreeFeatureSelector,
    CorrelationFeatureSelector,
    XGBGainSelector,
    StabilityNetSelector,
    RecencyFeatureSelector,
    FeatureSelector,
    build_selector,
)
