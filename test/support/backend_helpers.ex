defmodule NxPenalties.BackendHelpers do
  @moduledoc """
  Helpers for testing across multiple Nx backends.
  """

  @doc """
  Get list of available backends for testing.
  """
  def available_backends do
    backends = [{Nx.BinaryBackend, []}]

    # Check for EXLA
    backends =
      if Code.ensure_loaded?(EXLA.Backend) do
        [{EXLA.Backend, []} | backends]
      else
        backends
      end

    # Check for Torchx
    if Code.ensure_loaded?(Torchx.Backend) do
      [{Torchx.Backend, []} | backends]
    else
      backends
    end
  end

  @doc """
  Run test function on all available backends.
  """
  def with_backends(fun) do
    for {backend, opts} <- available_backends() do
      Nx.default_backend({backend, opts})

      try do
        fun.(backend)
      after
        Nx.default_backend(Nx.BinaryBackend)
      end
    end
  end

  @doc """
  Tag for tests that require EXLA.
  """
  def exla_tag do
    if Code.ensure_loaded?(EXLA.Backend), do: [], else: [:skip]
  end
end
