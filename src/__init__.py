"""Fasanara AI Credit Risk Analyst Challenge — solution package.

Modules
-------
config   Single source of truth: feature lists, CV protocol, decision-policy constants.
data     Loading, validation and joining of the provided CSVs.
model    Preprocessing + logistic-regression pipeline and its CV evaluation.
policy   Decision layer: cost-based alert threshold and Low/Medium/High risk ratings.
explain  Per-company analyst-style explanations grounded in the model's drivers.
"""
