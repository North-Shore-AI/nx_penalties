defmodule NxPenalties.MixProject do
  use Mix.Project

  @version "0.1.2"
  @source_url "https://github.com/North-Shore-AI/nx_penalties"
  @description "Composable regularization penalties and loss functions for the Nx ecosystem"

  def project do
    [
      app: :nx_penalties,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      description: @description,

      # Docs
      name: "NxPenalties",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs(),

      # Test
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.html": :test,
        "coveralls.json": :test
      ],
      elixirc_paths: elixirc_paths(Mix.env()),

      # Dialyzer
      dialyzer: [
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
        plt_add_apps: [:ex_unit],
        flags: [:unmatched_returns, :error_handling, :no_opaque]
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
      # Core
      {:nx, "~> 0.9"},
      {:nimble_options, "~> 1.0"},
      {:telemetry, "~> 1.0"},

      # Optional integrations
      {:axon, "~> 0.6", optional: true},
      {:polaris, "~> 0.1", optional: true},

      # Test
      {:exla, "~> 0.9", only: :test},
      {:stream_data, "~> 1.0", only: [:test, :dev]},
      {:excoveralls, "~> 0.18", only: :test},

      # Dev
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      name: "nx_penalties",
      maintainers: ["North-Shore-AI"],
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => "#{@source_url}/blob/main/CHANGELOG.md"
      },
      files: ~w(lib assets .formatter.exs mix.exs README.md LICENSE CHANGELOG.md)
    ]
  end

  defp docs do
    [
      main: "readme",
      assets: %{"assets" => "assets"},
      logo: "assets/nx_penalties.svg",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "LICENSE"
      ],
      source_ref: "v#{@version}",
      groups_for_modules: [
        "Core Penalties": [
          NxPenalties,
          NxPenalties.Penalties,
          NxPenalties.Divergences
        ],
        Pipeline: [
          NxPenalties.Pipeline
        ],
        Constraints: [
          NxPenalties.Constraints,
          NxPenalties.GradientTracker
        ],
        Integrations: [
          NxPenalties.Integration.Axon,
          NxPenalties.Integration.Polaris
        ],
        Telemetry: [
          NxPenalties.Telemetry
        ]
      ],
      groups_for_docs: [
        "Penalty Functions": &(&1[:section] == :penalties),
        "Divergence Functions": &(&1[:section] == :divergences),
        "Pipeline Operations": &(&1[:section] == :pipeline)
      ]
    ]
  end

  defp aliases do
    [
      quality: ["format", "credo --strict", "dialyzer"],
      "test.all": ["test --include integration"],
      setup: ["deps.get", "deps.compile"]
    ]
  end
end
