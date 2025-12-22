# "Accuracy is Not All You Need" - One-Page Summary

## Problem Statement

When Large Language Models (LLMs) are compressed using techniques such as quantization, the predominant evaluation method relies on measuring accuracy across various benchmarks.
If the compressed model's accuracy remains close to the baseline model, researchers assume negligible quality degradation. However, this paper challenges this assumption, arguing that **accuracy alone is insufficient** for comprehensive model evaluation, particularly for compressed LLMs.

## Key Arguments

### 1. **Limitations of Accuracy-Only Evaluation**

- Even when accuracy metrics remain stable post-compression, significant behavioral changes in model outputs may occur
- Traditional accuracy metrics fail to capture nuanced differences in model behavior, especially for tasks involving free-form text generation
- Models can maintain high accuracy while exhibiting different response patterns, confidence levels, and output distributions

### 2. **Proposed Alternative Metrics**

**KL-Divergence:**

- Measures the difference between probability distributions of the original and compressed models
- Captures changes in model confidence and output probability distributions
- Provides insight into how compression affects the model's internal decision-making process

**Flips Metric:**

- Novel metric that tracks when model predictions change from correct to incorrect (and vice versa) after compression
- Reveals instances where models change their answers even when overall accuracy remains high
- Particularly valuable for understanding model reliability and consistency

### 3. **Correlation Analysis**

The paper demonstrates that KL-Divergence and flips metrics are well-correlated, suggesting they capture similar underlying changes in model behavior that accuracy metrics miss.
