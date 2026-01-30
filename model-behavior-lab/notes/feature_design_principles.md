# Feature Design Principles

1. **Expose denominators**
   - Ratios like `appointment_success_rate` must be paired with `n_appointments` to avoid misleading signals.

2. **Avoid reusing rules as features**
   - Rule-based risk tiers should be used for operational triage, not as direct model inputs.

3. **Separate intervention signals**
   - Features that encode extra support (e.g., reminder intensity) must be isolated to prevent confounded interpretations.

4. **Prefer transparent transformations**
   - Simple, well-documented features are easier to explain and validate with domain experts.
