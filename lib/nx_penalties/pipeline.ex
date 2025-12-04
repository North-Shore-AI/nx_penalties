defmodule NxPenalties.Pipeline do
  @moduledoc """
  Composable pipeline for combining multiple penalties.

  Pipelines allow you to:
  - Combine multiple penalties with individual weights
  - Enable/disable penalties dynamically
  - Adjust weights during training (e.g., curriculum learning)
  - Collect metrics for each penalty

  ## Example

      pipeline =
        NxPenalties.Pipeline.new()
        |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 0.001)
        |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2, weight: 0.01)
        |> NxPenalties.Pipeline.add(:entropy, &NxPenalties.Divergences.entropy/2,
             weight: 0.1, opts: [mode: :penalty])

      {total_penalty, metrics} = NxPenalties.Pipeline.compute(pipeline, tensor)

  ## Entry Format

  Each entry is a 5-tuple: `{name, penalty_fn, weight, opts, enabled}`

    * `name` - Unique atom identifier for the penalty
    * `penalty_fn` - Function `(tensor, opts) -> scalar_tensor`
    * `weight` - Scaling factor (number or tensor)
    * `opts` - Options passed to the penalty function
    * `enabled` - Boolean to include/exclude from computation
  """

  defstruct entries: [], name: nil

  @type entry :: {atom(), function(), number() | Nx.Tensor.t(), keyword(), boolean()}
  @type t :: %__MODULE__{
          entries: [entry()],
          name: String.t() | nil
        }

  @doc """
  Create a new empty pipeline.

  ## Options

    * `:name` - Optional name for the pipeline (for logging/debugging)

  ## Examples

      pipeline = NxPenalties.Pipeline.new()
      pipeline = NxPenalties.Pipeline.new(name: "regularization")
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    name = Keyword.get(opts, :name, nil)
    %__MODULE__{entries: [], name: name}
  end

  @doc """
  Add a penalty to the pipeline.

  ## Parameters

    * `pipeline` - The pipeline struct
    * `name` - Unique atom identifier for this penalty
    * `penalty_fn` - Function with signature `(tensor, opts) -> scalar_tensor`
    * `opts` - Options:
      * `:weight` - Scaling factor. Default: `1.0`
      * `:opts` - Options passed to the penalty function. Default: `[]`
      * `:enabled` - Whether to include in computation. Default: `true`

  ## Examples

      pipeline
      |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)
      |> Pipeline.add(:l2, &Penalties.l2/2, weight: 0.001, opts: [clip: 1000.0])
  """
  @spec add(t(), atom(), function(), keyword()) :: t()
  def add(%__MODULE__{entries: entries} = pipeline, name, penalty_fn, opts \\ []) do
    weight = Keyword.get(opts, :weight, 1.0)
    penalty_opts = Keyword.get(opts, :opts, [])
    enabled = Keyword.get(opts, :enabled, true)

    entry = {name, penalty_fn, weight, penalty_opts, enabled}
    %{pipeline | entries: entries ++ [entry]}
  end

  @doc """
  Remove a penalty from the pipeline by name.

  ## Examples

      pipeline = Pipeline.remove(pipeline, :l1)
  """
  @spec remove(t(), atom()) :: t()
  def remove(%__MODULE__{entries: entries} = pipeline, name) do
    new_entries = Enum.reject(entries, fn {n, _, _, _, _} -> n == name end)
    %{pipeline | entries: new_entries}
  end

  @doc """
  Update the weight of a penalty.

  ## Examples

      pipeline = Pipeline.update_weight(pipeline, :l1, 0.05)
      pipeline = Pipeline.update_weight(pipeline, :l1, Nx.tensor(0.05))
  """
  @spec update_weight(t(), atom(), number() | Nx.Tensor.t()) :: t()
  def update_weight(%__MODULE__{entries: entries} = pipeline, name, new_weight) do
    new_entries =
      Enum.map(entries, fn
        {^name, fn_, _weight, opts, enabled} -> {name, fn_, new_weight, opts, enabled}
        entry -> entry
      end)

    %{pipeline | entries: new_entries}
  end

  @doc """
  Enable or disable a penalty.

  ## Examples

      pipeline = Pipeline.set_enabled(pipeline, :l1, false)
  """
  @spec set_enabled(t(), atom(), boolean()) :: t()
  def set_enabled(%__MODULE__{entries: entries} = pipeline, name, enabled) do
    new_entries =
      Enum.map(entries, fn
        {^name, fn_, weight, opts, _enabled} -> {name, fn_, weight, opts, enabled}
        entry -> entry
      end)

    %{pipeline | entries: new_entries}
  end

  @doc """
  Compute all penalties and return total with metrics.

  ## Parameters

    * `pipeline` - The pipeline struct
    * `tensor` - Input tensor to penalize
    * `opts` - Options:
      * `:extra_args` - Additional args merged into each penalty's opts
      * `:track_grad_norms` - Compute gradient norms (default: `false`)

  ## Returns

  A tuple `{total, metrics}` where:
    * `total` - Scalar tensor with sum of weighted penalties
    * `metrics` - Map with per-penalty and aggregate metrics

  ## Metrics Format

      %{
        "l1" => 6.0,              # Raw penalty value
        "l1_weighted" => 0.06,    # After weight applied
        "l2" => 14.0,
        "l2_weighted" => 0.014,
        "total" => 0.074          # Sum of weighted
      }

  ## Examples

      {total, metrics} = Pipeline.compute(pipeline, tensor)
      {total, metrics} = Pipeline.compute(pipeline, tensor, extra_args: [reduction: :mean])
  """
  @spec compute(t(), Nx.Tensor.t(), keyword()) :: {Nx.Tensor.t(), map()}
  def compute(%__MODULE__{} = pipeline, tensor, opts \\ []) do
    NxPenalties.Telemetry.span_pipeline(pipeline, tensor, opts, &do_compute/3)
  end

  defp do_compute(%__MODULE__{entries: entries} = pipeline, tensor, opts) do
    extra_args = Keyword.get(opts, :extra_args, [])
    track_grad_norms = Keyword.get(opts, :track_grad_norms, false)

    # Filter enabled entries
    enabled_entries = Enum.filter(entries, fn {_, _, _, _, enabled} -> enabled end)

    if enabled_entries == [] do
      {Nx.tensor(0.0), %{}}
    else
      # Compute each penalty
      {total, metrics} =
        Enum.reduce(enabled_entries, {Nx.tensor(0.0), %{}}, fn
          {name, penalty_fn, weight, penalty_opts, _enabled}, {acc_total, acc_metrics} ->
            # Merge extra_args with penalty opts (extra_args take precedence)
            merged_opts = Keyword.merge(penalty_opts, extra_args)

            # Compute raw penalty
            raw_penalty = penalty_fn.(tensor, merged_opts)
            raw_value = Nx.to_number(raw_penalty)

            # Apply weight
            weighted_penalty = Nx.multiply(raw_penalty, weight)
            weighted_value = Nx.to_number(weighted_penalty)

            # Update metrics
            name_str = Atom.to_string(name)

            new_metrics =
              acc_metrics
              |> Map.put(name_str, raw_value)
              |> Map.put("#{name_str}_weighted", weighted_value)

            # Accumulate total
            new_total = Nx.add(acc_total, weighted_penalty)

            {new_total, new_metrics}
        end)

      # Add total to metrics
      total_value = Nx.to_number(total)
      metrics = Map.put(metrics, "total", total_value)

      # Optionally add gradient metrics
      metrics =
        if track_grad_norms do
          grad_metrics = NxPenalties.GradientTracker.pipeline_grad_norms(pipeline, tensor)
          total_norm = NxPenalties.GradientTracker.total_grad_norm(pipeline, tensor)

          metrics
          |> Map.merge(grad_metrics)
          |> Map.put("total_grad_norm", total_norm)
        else
          metrics
        end

      {total, metrics}
    end
  end

  @doc """
  Compute only the total penalty (gradient-compatible).

  Unlike `compute/3`, this function does NOT call `Nx.to_number` and
  is therefore compatible with `Nx.Defn.grad/1`.

  ## Examples

      # Can be used with gradients
      grad_fn = Nx.Defn.grad(fn tensor ->
        Pipeline.compute_total(pipeline, tensor)
      end)
  """
  @spec compute_total(t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def compute_total(%__MODULE__{entries: entries}, tensor, opts \\ []) do
    extra_args = Keyword.get(opts, :extra_args, [])

    # Filter enabled entries
    enabled_entries = Enum.filter(entries, fn {_, _, _, _, enabled} -> enabled end)

    if enabled_entries == [] do
      Nx.tensor(0.0)
    else
      Enum.reduce(enabled_entries, Nx.tensor(0.0), fn
        {_name, penalty_fn, weight, penalty_opts, _enabled}, acc_total ->
          merged_opts = Keyword.merge(penalty_opts, extra_args)
          raw_penalty = penalty_fn.(tensor, merged_opts)
          weighted_penalty = Nx.multiply(raw_penalty, weight)
          Nx.add(acc_total, weighted_penalty)
      end)
    end
  end
end
