defmodule NxPenalties.Penalties.Validation do
  @moduledoc false

  @l1_schema [
    lambda: [type: {:or, [:float, :integer]}, default: 1.0],
    reduction: [type: {:in, [:sum, :mean]}, default: :sum]
  ]

  @l2_schema [
    lambda: [type: {:or, [:float, :integer]}, default: 1.0],
    reduction: [type: {:in, [:sum, :mean]}, default: :sum],
    clip: [type: {:or, [:float, :integer, nil]}, default: nil]
  ]

  @elastic_net_schema [
    lambda: [type: {:or, [:float, :integer]}, default: 1.0],
    l1_ratio: [type: {:or, [:float, :integer]}, default: 0.5],
    reduction: [type: {:in, [:sum, :mean]}, default: :sum]
  ]

  @doc false
  def validate_l1!(opts), do: NimbleOptions.validate!(opts, @l1_schema)

  @doc false
  def validate_l2!(opts), do: NimbleOptions.validate!(opts, @l2_schema)

  @doc false
  def validate_elastic_net!(opts), do: NimbleOptions.validate!(opts, @elastic_net_schema)
end
