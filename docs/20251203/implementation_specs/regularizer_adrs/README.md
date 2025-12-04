# Regularizer ADRs

Architecture Decision Records for additional regularizers in Tinkex.

## Context

The `Tinkex.Regularizer` behaviour provides a composable pipeline for adding penalty terms to the training loss. This ADR set documents proposed regularizers for the library.

## ADR Index

| ADR | Title | Status | Priority |
|-----|-------|--------|----------|
| [ADR-001](adrs/ADR-001_l1_sparsity.md) | L1 Sparsity Regularizer | Proposed | High |
| [ADR-002](adrs/ADR-002_l2_weight_decay.md) | L2 Weight Decay Regularizer | Proposed | High |
| [ADR-003](adrs/ADR-003_elastic_net.md) | Elastic Net Regularizer | Proposed | Medium |
| [ADR-004](adrs/ADR-004_kl_divergence.md) | KL Divergence Regularizer | Proposed | High |
| [ADR-005](adrs/ADR-005_entropy.md) | Entropy Regularizer | Proposed | Medium |
| [ADR-006](adrs/ADR-006_consistency.md) | Consistency Regularizer | Proposed | Medium |
| [ADR-007](adrs/ADR-007_gradient_penalty.md) | Gradient Penalty Regularizer | Proposed | Low |
| [ADR-008](adrs/ADR-008_orthogonality.md) | Orthogonality Regularizer | Proposed | Medium |

## Regularizer Categories

### Norm-Based (ADR-001, ADR-002, ADR-003)
Classical regularizers that penalize parameter magnitudes. Prevent overfitting by constraining model capacity.

### Distribution-Based (ADR-004, ADR-005)
Regularizers that operate on output distributions. Useful for keeping fine-tuned models close to base models or controlling prediction confidence.

### Stability-Based (ADR-006, ADR-007)
Regularizers that encourage consistent or smooth model behavior. Improve generalization and robustness.

### Structure-Based (ADR-008)
Regularizers that enforce structural properties on learned representations. Particularly relevant for LoRA fine-tuning.

## Implementation Priority

### Phase 1 (Core)
- L1, L2, KL Divergence - fundamental regularizers used in most training scenarios

### Phase 2 (Extended)
- Elastic Net, Entropy, Orthogonality - common but more specialized use cases

### Phase 3 (Advanced)
- Consistency, Gradient Penalty - advanced techniques for specific scenarios

## Common Interface

All regularizers implement the `Tinkex.Regularizer` behaviour:

```elixir
@callback compute(
  data :: list(Datum.t()),
  logprobs :: Nx.Tensor.t(),
  opts :: keyword()
) :: {Nx.Tensor.t(), %{String.t() => number()}}

@callback name() :: String.t()
```

## Usage Pattern

```elixir
alias Tinkex.Regularizer.{L1, L2, KLDivergence}
alias Tinkex.Types.RegularizerSpec

specs = [
  %RegularizerSpec{fn: &L1.compute/3, weight: 0.001, name: "l1"},
  %RegularizerSpec{fn: &L2.compute/3, weight: 0.01, name: "l2"},
  %RegularizerSpec{fn: &KLDivergence.compute/3, weight: 0.1, name: "kl_div"}
]

{:ok, outputs} = Tinkex.Regularizer.Executor.execute_all(specs, data, logprobs, [])
```
