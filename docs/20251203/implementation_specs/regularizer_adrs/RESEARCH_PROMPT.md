# Research Prompt: Regularizer Library Placement in Nx Ecosystem

## Objective

Determine the optimal home for a composable regularizer/loss penalty library within the Elixir Nx ecosystem. The options are:

1. **Contribute to Axon** - Add regularizer infrastructure to the existing neural network library
2. **Create standalone Nx extension** - New package like `nx_regularizers` or `nx_losses`
3. **Contribute to Polaris** - Axon's optimizer/loss library
4. **Other** - Discover if there's a better home

## Context

We have implemented a regularizer system with:

- **Behaviour-based interface**: `compute(data, logprobs, opts) -> {loss_tensor, metrics}`
- **Executor with parallelism**: `Task.async_stream` for concurrent regularizer evaluation
- **Pipeline composition**: Multiple regularizers combined with weights
- **Gradient tracking**: Optional gradient norm computation via `Nx.Defn.grad`
- **Telemetry integration**: Instrumentation for monitoring

### Regularizers Designed

| Regularizer | Purpose | Complexity |
|-------------|---------|------------|
| L1 (Lasso) | Sparsity | Simple |
| L2 (Ridge) | Smoothing | Simple |
| Elastic Net | L1+L2 combo | Simple |
| KL Divergence | Distribution matching | Medium |
| Entropy | Confidence control | Medium |
| Consistency | Perturbation stability | Medium |
| Gradient Penalty | Lipschitz smoothness | Complex |
| Orthogonality | Representation diversity | Complex |

### Current Implementation

```elixir
defmodule Tinkex.Regularizer do
  @callback compute(
    data :: list(map()),
    logprobs :: Nx.Tensor.t(),
    opts :: keyword()
  ) :: {Nx.Tensor.t(), %{String.t() => number()}}

  @callback name() :: String.t()
end
```

The regularizers are pure Nx tensor operations with no Tinkex-specific dependencies.

---

## Research Questions

### Part 1: Axon Deep Dive

#### 1.1 Current Loss Architecture
- How does `Axon.Losses` work internally?
- What loss functions are currently implemented?
- How are custom losses integrated into training loops?
- Does Axon support loss composition (multiple loss terms)?

#### 1.2 Regularization in Axon
- Does Axon have any regularization support currently?
- How do `Axon.Layers` handle activity regularization?
- Is there a `regularizers` option on layers like Keras has?
- What's the current pattern for adding L1/L2 penalties?

#### 1.3 Training Loop Integration
- How does `Axon.Loop` consume losses?
- Can losses return auxiliary metrics (like our `metrics` map)?
- How would a regularizer pipeline integrate with `Axon.Loop.trainer/3`?
- What hooks/callbacks exist for injecting custom loss terms?

#### 1.4 Axon's Design Philosophy
- Read Axon's CONTRIBUTING.md or design docs
- What's their stance on expanding scope?
- Are there open issues/PRs related to regularization?
- Who are the maintainers and what's their responsiveness?

### Part 2: Polaris Investigation

#### 2.1 What is Polaris?
- Understand Polaris's relationship to Axon
- What does it currently contain? (optimizers? losses?)
- Is it meant to be Axon-specific or general Nx utilities?

#### 2.2 Polaris Scope
- Would regularizers fit Polaris's mission?
- Are there precedents for loss-related additions to Polaris?
- Check Polaris issues/discussions for relevant context

### Part 3: Nx Extension Package Patterns

#### 3.1 Existing Extensions Analysis
Study these packages for patterns:

| Package | Purpose | Relationship to Nx |
|---------|---------|-------------------|
| Scholar | Traditional ML | Uses Nx, independent |
| Polaris | Optimizers | Axon companion |
| Bumblebee | Transformers | Uses Axon + Nx |
| Ortex | ONNX runtime | Nx backend |
| Tokenizers | Text tokenization | NIF, Nx-adjacent |
| ExLA | XLA backend | Nx backend |

For each, investigate:
- How do they structure their API?
- What's their dependency relationship with Nx?
- Do they use `Nx.Defn` for JIT compilation?
- How do they handle Nx backend compatibility?
- What's their versioning strategy relative to Nx?

#### 3.2 Package Structure Conventions
- Standard directory layout for Nx ecosystem packages
- Naming conventions (`Nx.*` vs standalone namespace)
- Documentation patterns (livebook examples, hexdocs structure)
- Test patterns (how to test across backends)

#### 3.3 Backend Compatibility
- How do Nx extensions ensure they work across backends (EXLA, Torchx, default)?
- Are there Nx features that don't work on all backends?
- How to write backend-agnostic code?
- Testing matrix requirements

### Part 4: Creating an Nx Extension Package

#### 4.1 Technical Requirements
- Minimum Nx version to depend on
- Required vs optional dependencies
- How to properly declare Nx backend compatibility
- CI/CD setup for multi-backend testing

#### 4.2 Naming & Namespace
Evaluate options:
- `nx_regularizers` - Nx ecosystem branding
- `nx_losses` - Broader scope (losses + regularizers)
- `nx_penalties` - Alternative framing
- `scholar_contrib` - Community extensions to Scholar
- Custom namespace (e.g., `Penumbra`, `Gradient`) - independent branding

#### 4.3 API Design Considerations
- Should it use `Nx.Defn` for JIT compilation?
- Stateless functions vs behaviour-based modules?
- How to handle regularizers that need external state (e.g., KL divergence needs reference distribution)?
- Integration patterns for Axon users vs raw Nx users

#### 4.4 Publishing & Maintenance
- Hex.pm publishing process
- Documentation requirements
- Versioning strategy
- Community building (should it be under elixir-nx org?)

### Part 5: Decision Framework

#### 5.1 Axon Contribution Path
If recommending Axon contribution:
- What would the PR look like?
- Which modules would be added/modified?
- Expected timeline for review/merge?
- Risk of rejection or scope disagreement?

#### 5.2 Standalone Package Path
If recommending standalone package:
- Proposed package name and namespace
- Dependency graph
- MVP scope (which regularizers for v0.1?)
- Long-term maintenance considerations

#### 5.3 Hybrid Approach
Could we:
- Create standalone package first
- Propose Axon integration that depends on it?
- Or contribute core to Axon, extras as extension?

---

## Specific Files/Resources to Examine

### Axon Codebase
```
axon/
├── lib/axon/
│   ├── losses.ex          # Current loss implementations
│   ├── loop.ex            # Training loop
│   ├── layers.ex          # Layer definitions (regularization hooks?)
│   └── compiler.ex        # How models are compiled
├── CONTRIBUTING.md
└── mix.exs                # Dependencies, version constraints
```

### Polaris Codebase
```
polaris/
├── lib/polaris/
│   ├── optimizers.ex      # Optimizer implementations
│   └── updates.ex         # Gradient update functions
└── mix.exs
```

### Scholar Codebase (for patterns)
```
scholar/
├── lib/scholar/
│   └── linear/
│       ├── ridge_regression.ex    # L2 regularization example
│       └── logistic_regression.ex # May have regularization
└── mix.exs
```

### GitHub Issues/Discussions to Find
- Any Axon issues mentioning "regularization", "L1", "L2", "activity_regularizer"
- Any Nx discussions about loss function libraries
- Scholar discussions about scope expansion
- Elixir Forum threads on regularization in Nx ecosystem

---

## Expected Deliverables

### 1. Architecture Analysis
- Diagram of Nx ecosystem package relationships
- Where regularizers fit architecturally
- Dependencies and integration points

### 2. Recommendation with Rationale
- Primary recommendation (Axon, Polaris, standalone, or other)
- Backup recommendation if primary is rejected
- Clear pros/cons for each option

### 3. Implementation Roadmap
If standalone package:
- Package name recommendation
- MVP feature set
- Directory structure
- Sample mix.exs with dependencies
- Sample module structure

If Axon/Polaris contribution:
- Proposed module locations
- API changes needed
- Draft PR description

### 4. Risk Assessment
- Maintenance burden for each option
- Community adoption likelihood
- Compatibility risks with Nx version changes
- Bus factor / sustainability concerns

### 5. Code Samples
- Example of how each regularizer would look in recommended location
- Integration example showing usage in training loop
- Test examples showing backend compatibility approach

---

## Success Criteria

The research is complete when we can confidently answer:

1. **Where should the regularizers live?** (specific package/location)
2. **What should the API look like?** (function signatures, composition patterns)
3. **What's the MVP scope?** (which regularizers, what features)
4. **What's the path to implementation?** (PR vs new package, steps involved)
5. **What are the maintenance implications?** (dependencies, testing, versioning)

---

## Notes for Research Agent

- Prioritize primary sources (actual code, official docs) over blog posts
- Check GitHub issues and PRs for maintainer opinions on scope
- Look for prior art - has anyone attempted this before?
- Consider the Elixir community culture around package proliferation
- The Nx ecosystem is still maturing - what's the trajectory?
- José Valim is heavily involved in Nx - any statements on ecosystem organization?
