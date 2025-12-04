# Configure ExUnit
ExUnit.start(exclude: [:skip])

# Set default backend for tests
Nx.default_backend(Nx.BinaryBackend)

# Optional: Configure EXLA for GPU tests
if Code.ensure_loaded?(EXLA) do
  Application.put_env(:exla, :clients,
    cuda: [platform: :cuda],
    rocm: [platform: :rocm],
    tpu: [platform: :tpu],
    host: [platform: :host]
  )
end
