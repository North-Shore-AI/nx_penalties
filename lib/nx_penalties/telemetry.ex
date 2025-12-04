defmodule NxPenalties.Telemetry do
  @moduledoc """
  Telemetry events for NxPenalties operations.

  ## Events

  ### `[:nx_penalties, :penalty, :compute, :start]`

  Emitted before computing a single penalty.

  Measurements: `%{system_time: integer}`
  Metadata: `%{name: atom, opts: keyword}`

  ### `[:nx_penalties, :penalty, :compute, :stop]`

  Emitted after computing a single penalty.

  Measurements: `%{duration: integer, value: float}`
  Metadata: `%{name: atom, opts: keyword}`

  ### `[:nx_penalties, :pipeline, :compute, :start]`

  Emitted before computing a pipeline.

  Measurements: `%{system_time: integer}`
  Metadata: `%{pipeline_name: String.t | nil, entry_count: integer}`

  ### `[:nx_penalties, :pipeline, :compute, :stop]`

  Emitted after computing a pipeline.

  Measurements: `%{duration: integer, total: float}`
  Metadata: `%{pipeline_name: String.t | nil, entry_count: integer, metrics: map}`

  ## Example Handler

      :telemetry.attach_many(
        "nx-penalties-logger",
        [
          [:nx_penalties, :pipeline, :compute, :stop]
        ],
        fn event, measurements, metadata, _config ->
          IO.puts("Pipeline computed in \#{measurements.duration}ns, total: \#{measurements.total}")
        end,
        nil
      )
  """

  @doc """
  Execute a pipeline computation with telemetry events.
  """
  @spec span_pipeline(NxPenalties.Pipeline.t(), Nx.Tensor.t(), keyword(), function()) ::
          {Nx.Tensor.t(), map()}
  def span_pipeline(pipeline, tensor, opts, compute_fn) do
    metadata = %{
      pipeline_name: pipeline.name,
      entry_count: length(pipeline.entries)
    }

    :telemetry.span(
      [:nx_penalties, :pipeline, :compute],
      metadata,
      fn ->
        {total, metrics} = compute_fn.(pipeline, tensor, opts)
        total_value = Nx.to_number(total)

        result_metadata =
          Map.merge(metadata, %{
            metrics: metrics,
            total: total_value
          })

        {{total, metrics}, result_metadata}
      end
    )
  end

  @doc """
  Emit a penalty computation event.
  """
  @spec emit_penalty_computed(atom(), float(), integer()) :: :ok
  def emit_penalty_computed(name, value, duration_ns) do
    :telemetry.execute(
      [:nx_penalties, :penalty, :computed],
      %{value: value, duration: duration_ns},
      %{name: name}
    )
  end
end
