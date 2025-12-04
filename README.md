<p align="center">
  <img src="assets/nx_penalties.svg" alt="NxPenalties" width="400">
</p>

# NxPenalties

Regularization and penalty functions for Elixir Nx.

## Overview

NxPenalties provides composable regularization functions for machine learning with [Nx](https://github.com/elixir-nx/nx):

- **L1 (Lasso)**: Sparse weight regularization via absolute values
- **L2 (Ridge)**: Weight decay via squared norms
- **Elastic Net**: Combined L1+L2 regularization
- **Custom penalties**: Extensible penalty function API

## Installation

```elixir
def deps do
  [
    {:nx_penalties, "~> 0.1.0"}
  ]
end
```

## Usage

```elixir
# L1 regularization
l1_penalty = NxPenalties.l1(weights, lambda: 0.01)

# L2 regularization
l2_penalty = NxPenalties.l2(weights, lambda: 0.001)

# Elastic net (combined)
elastic_penalty = NxPenalties.elastic_net(weights, l1_ratio: 0.5, lambda: 0.01)

# Add to loss
total_loss = Nx.add(base_loss, penalty)
```

## License

MIT
