#!/bin/bash
# Run all NxPenalties examples
#
# Usage: ./examples/run_all.sh

set -e

cd "$(dirname "$0")/.."

echo "========================================"
echo "  Running NxPenalties Examples"
echo "========================================"
echo ""

examples=(
  "basic_usage.exs"
  "pipeline_composition.exs"
  "axon_training.exs"
  "curriculum_learning.exs"
  "gradient_tracking.exs"
  "polaris_integration.exs"
  "constraints.exs"
  "entropy_normalization.exs"
  "gradient_penalty.exs"
)

for example in "${examples[@]}"; do
  echo "----------------------------------------"
  echo "Running: $example"
  echo "----------------------------------------"
  mix run "examples/$example"
  echo ""
done

echo "========================================"
echo "  All examples completed!"
echo "========================================"
