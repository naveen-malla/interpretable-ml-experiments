# Explanation Failure Modes

This note captures failure patterns observed when explanations are generated from models trained on ratio features and rule-based proxies.

## Common Failure Modes

- **Paradoxical success signals**: High appointment success rates can correlate with elevated no-show risk when the ratio omits the denominator (e.g., a perfect record from very few visits).
- **Proxy leakage**: Reusing rule-based risk tiers as model inputs can cause the model to explain itself with the proxy instead of the underlying behaviors.
- **Implicit intervention confounding**: When high-risk patients receive additional reminders, their improved outcomes can make the model infer that risk indicators are protective.

## Mitigation Strategies

- Prefer count-based features with explicit denominators.
- Separate risk flags used for intervention from model inputs used for prediction.
- Apply monotonic constraints to adherence indicators to preserve semantic correctness.
