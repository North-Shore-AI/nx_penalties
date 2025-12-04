defmodule NxPenalties.GradientTracker do
  @moduledoc """
  Computes gradient norms for regularizers using Nx automatic differentiation.

  > **v0.2 Preview**: Basic gradient norm computation available.
  > Full pipeline integration coming in v0.2.

  ## Purpose

  When training with multiple regularizers, it's crucial to understand which
  penalties contribute most to the gradient signal. This module provides:

  - Per-penalty gradient norms
  - Total composed gradient norm
  - Gradient ratio analysis

  ## Usage

  Enable in pipeline computation (v0.2):

      {total, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: true)

      metrics["l1_grad_norm"]      # L2 norm of L1 penalty's gradient
      metrics["total_grad_norm"]   # Combined gradient norm

  ## Performance Note

  Gradient tracking requires additional backward passes. Only enable when
  debugging or for periodic monitoring (e.g., every 100 steps).

  ## Important: What Are We Differentiating?

  These functions compute ∂penalty/∂(pipeline_input), NOT ∂penalty/∂params.

  The "pipeline input" is whatever tensor you pass to `Pipeline.compute/3`—
  typically model outputs, activations, or logprobs. This tells you how
  sensitive each penalty is to changes in that tensor.
  """

  require Logger

  @doc """
  Compute L2 norm of gradients from a loss function.

  ## Parameters

    * `loss_fn` - Function `(tensor) -> scalar_tensor`
    * `tensor` - Input to differentiate with respect to

  ## Returns

  Float representing L2 norm: `||∇f||₂ = sqrt(Σ grad²)`
  Returns `nil` if gradient computation fails.

  ## Examples

      iex> loss_fn = fn x -> Nx.sum(Nx.abs(x)) end
      iex> GradientTracker.compute_grad_norm(loss_fn, Nx.tensor([1.0, -2.0, 3.0]))
      1.732...  # sqrt(3) since ∂|x|/∂x = sign(x) = [1, -1, 1]

      iex> loss_fn = fn x -> Nx.sum(Nx.pow(x, 2)) end
      iex> GradientTracker.compute_grad_norm(loss_fn, Nx.tensor([1.0, 2.0, 3.0]))
      7.483...  # sqrt(4 + 16 + 36) since ∂x²/∂x = 2x
  """
  @spec compute_grad_norm((Nx.Tensor.t() -> Nx.Tensor.t()), Nx.Tensor.t()) :: float() | nil
  def compute_grad_norm(loss_fn, tensor) do
    grad_fn = Nx.Defn.grad(loss_fn)
    grad_tensor = grad_fn.(tensor)

    grad_tensor
    |> Nx.flatten()
    |> Nx.pow(2)
    |> Nx.sum()
    |> Nx.sqrt()
    |> Nx.to_number()
  rescue
    e ->
      Logger.warning("""
      Gradient computation failed: #{inspect(e)}

      This usually means the loss function contains non-differentiable operations.
      Consider disabling gradient tracking or fixing the penalty function.
      """)

      nil
  end

  @doc """
  Compute gradient norms for all penalties in a pipeline.

  ## Parameters

    * `pipeline` - Pipeline struct with penalty entries
    * `tensor` - Input tensor to differentiate with respect to

  ## Returns

  Map of `%{"name_grad_norm" => float, ...}`

  ## Examples

      pipeline = NxPenalties.pipeline([
        {:l1, weight: 0.001},
        {:l2, weight: 0.01}
      ])

      norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)
      # %{"l1_grad_norm" => 1.732, "l2_grad_norm" => 7.483}
  """
  @spec pipeline_grad_norms(NxPenalties.Pipeline.t(), Nx.Tensor.t()) :: map()
  def pipeline_grad_norms(pipeline, tensor) do
    pipeline.entries
    |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
    |> Enum.flat_map(fn {name, penalty_fn, _weight, opts, _enabled} ->
      loss_fn = fn t -> penalty_fn.(t, opts) end

      case compute_grad_norm(loss_fn, tensor) do
        nil ->
          [{"#{name}_grad_norm", nil}, {"#{name}_grad_norm_error", true}]

        norm ->
          [{"#{name}_grad_norm", norm}]
      end
    end)
    |> Map.new()
  end

  @doc """
  Compute gradient norm for the total weighted pipeline loss.

  ## Formula

      total = Σ(weight_i × penalty_i(tensor))
      result = ||∇_tensor total||₂

  ## Parameters

    * `pipeline` - Pipeline struct
    * `tensor` - Input tensor

  ## Returns

  Float representing total gradient norm, or `nil` on failure.

  ## Examples

      norm = GradientTracker.total_grad_norm(pipeline, tensor)
  """
  @spec total_grad_norm(NxPenalties.Pipeline.t(), Nx.Tensor.t()) :: float() | nil
  def total_grad_norm(pipeline, tensor) do
    total_loss_fn = fn t ->
      pipeline.entries
      |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
      |> Enum.map(fn {_name, penalty_fn, weight, opts, _enabled} ->
        Nx.multiply(penalty_fn.(t, opts), weight)
      end)
      |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
    end

    compute_grad_norm(total_loss_fn, tensor)
  end
end
