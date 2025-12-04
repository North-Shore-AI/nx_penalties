# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### KL Divergence Options (ADR-010)
- `:direction` option for `kl_divergence/3` - Choose `:forward` (KL(P||Q), default) or `:reverse` (KL(Q||P))
- `:symmetric` option for `kl_divergence/3` - Compute 0.5 * (KL(P||Q) + KL(Q||P)) when `true`

#### Entropy Temperature (ADR-011)
- `:temperature` option for `entropy/2` - Scale distribution sharpness:
  - `< 1.0` - Sharper distribution (lower entropy)
  - `1.0` - No scaling (default)
  - `> 1.0` - Flatter distribution (higher entropy)

#### Pipeline.Multi (ADR-012)
- `NxPenalties.Pipeline.Multi` module for multi-input penalty composition
- Named input support for divergence and consistency penalties
- Weight updates and enable/disable per-penalty
- Gradient tracking for multi-input pipelines

#### Top-Level Delegates
- `NxPenalties.orthogonality/2` - Delegate to `Constraints.orthogonality/2`
- `NxPenalties.consistency/3` - Delegate to `Constraints.consistency/3`

#### Pipeline Mappings
- `:orthogonality` support in `NxPenalties.pipeline/1`
- `:output_magnitude` support in `NxPenalties.pipeline/1`

#### Property-Based Tests
- Property tests for L1/L2/Elastic Net penalties (non-negativity, scaling, symmetry)
- Property tests for KL/JS divergence and entropy (bounds, symmetry)
- StreamData-based generators for tensors and distributions

#### CI Improvements
- Multi-backend test matrix (Binary, EXLA CPU)
- Separate property test job
- Code format check job

### Changed
- L2 validation schema now includes `:center` option (nil | :mean | number)

## [0.1.2] - 2025-12-05

### Fixed
- Axon/Polaris optional deps are now truly optional: integration modules compile only when those libraries are present and provide clear error messages otherwise, preventing downstream projects without Axon/Polaris from failing to compile.

## [0.1.1] - 2025-12-04

### Added

#### Pipeline
- `differentiable: false` option in `Pipeline.add/4` - Skip gradient tracking for penalties containing non-differentiable operations like `Nx.argmax/2`
- `meta` field on Pipeline struct for per-penalty metadata

#### Axon Integration
- `build_train_step/4` - Custom training step with full metrics access from penalty pipeline
- `build_train_step_with_weight_decay/5` - Training step with L2 penalty on model parameters
- `capture_activation/2` - Insert capture layer to store intermediate activations
- `extract_captures/1` - Extract captured activations from model state
- `build_activity_train_step/4` - Training step with activity regularization on captured layers
- `weight_schedule/1` - Create curriculum learning weight schedules (linear, warmup, decay, constant)
- `apply_scheduled_weights/2` - Apply scheduled weights to a pipeline
- `flatten_params/1` - Flatten nested parameter maps to tensor list
- `log_penalties/3` - Add penalty metrics logging to Axon.Loop
- `schedule_weights/4` - Add dynamic weight scheduling callback to Axon.Loop
- `trainer/5` - Convenience function to create training loop with integrated penalties

#### Polaris Integration
- `add_gradient_clipping/2` - Global gradient norm clipping to prevent explosion
- `add_gradient_noise/3` - Decaying Gaussian noise for regularization (Neelakantan et al., 2015)
- `add_adaptive_gradient_clipping/3` - Per-parameter AGC (Brock et al., 2021)
- `add_gradient_centralization/2` - Zero-mean gradients for stability (Yong et al., 2020)

#### Infrastructure
- GitHub Actions CI workflow for tests and Dialyzer

#### Examples
- `axon_full_integration.exs` - Comprehensive 8-part Axon integration demo
- `polaris_full_integration.exs` - Comprehensive 10-part Polaris integration demo

### Fixed
- Dialyzer errors for Axon.Loop.t/0 unknown type (replaced with struct())

## [0.1.0] - 2025-12-03

Initial release.

[Unreleased]: https://github.com/North-Shore-AI/nx_penalties/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/North-Shore-AI/nx_penalties/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/North-Shore-AI/nx_penalties/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/North-Shore-AI/nx_penalties/releases/tag/v0.1.0
