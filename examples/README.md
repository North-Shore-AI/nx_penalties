# NxPenalties Examples

Examples demonstrating NxPenalties regularization features.

## Running Examples

Run individual examples:

```bash
mix run examples/basic_usage.exs
mix run examples/pipeline_composition.exs
mix run examples/axon_training.exs
mix run examples/curriculum_learning.exs
mix run examples/gradient_tracking.exs
mix run examples/polaris_integration.exs
mix run examples/constraints.exs
mix run examples/entropy_normalization.exs
mix run examples/gradient_penalty.exs
```

Run all examples:

```bash
./examples/run_all.sh
```

## Examples

### basic_usage.exs

Core penalty functions: L1, L2, and Elastic Net with different lambda values and reduction modes.

### pipeline_composition.exs

Building penalty pipelines with multiple regularizers, dynamic weight adjustment, enable/disable controls, and gradient-compatible computation.

### axon_training.exs

Integration with Axon neural networks. Shows how to add regularization to training loops.

Requires Axon (`{:axon, "~> 0.6"}`) - falls back to conceptual example if unavailable.

### curriculum_learning.exs

Dynamic penalty scheduling for curriculum learning:
- Decreasing regularization over epochs
- Phase-based training with different penalty configurations
- Elastic Net ratio shifting
- Gradient flow verification

### gradient_tracking.exs

Monitor gradient norms from regularization penalties:
- Enable `track_grad_norms: true` in pipeline compute
- Per-penalty gradient norm metrics
- Direct GradientTracker usage
- Tensor validation with `NxPenalties.validate/1`

### polaris_integration.exs

Gradient-level weight decay transforms for Polaris optimizers:
- L2 weight decay (AdamW-style)
- L1 weight decay (sparsity)
- Elastic Net decay
- Composing multiple transforms

### constraints.exs

Structural constraint penalties:
- Orthogonality penalty for decorrelating representations
- Consistency penalty for paired output stability
- Different metrics (MSE, L1, cosine)
- Soft vs hard modes

### entropy_normalization.exs

Entropy regularization with optional normalization:
- Bonus vs penalty modes
- Normalized entropy scaled to [0, 1]
- Impact on pipelines and metrics

### gradient_penalty.exs

Gradient penalty primitives and cheaper proxies:
- Full gradient norm penalty via `gradient_penalty/3`
- WGAN-GP style interpolated penalty
- Output magnitude proxy for performance
