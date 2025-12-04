defmodule NxPenalties.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/nx_penalties"

  def project do
    [
      app: :nx_penalties,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "NxPenalties",
      source_url: @source_url,
      docs: docs(),

      # Test
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.html": :test
      ],
      elixirc_paths: elixirc_paths(Mix.env()),

      # Dialyzer
      dialyzer: [
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
        plt_add_apps: [:ex_unit]
      ],

      # Aliases
      aliases: aliases()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Core - use local Nx source
      {:nx, path: "nx/nx", override: true},
      {:nimble_options, "~> 1.0"},
      {:telemetry, "~> 1.0"},

      # Optional integrations
      {:axon, "~> 0.6", optional: true},
      {:polaris, "~> 0.1", optional: true},

      # Test - use local EXLA source
      {:exla, path: "nx/exla", only: :test},
      {:stream_data, "~> 1.0", only: [:test, :dev]},
      {:excoveralls, "~> 0.18", only: :test},

      # Dev
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md"],
      source_ref: "v#{@version}"
    ]
  end

  defp aliases do
    [
      quality: ["format", "credo --strict", "dialyzer"]
    ]
  end
end
