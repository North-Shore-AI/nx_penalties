if Code.ensure_loaded?(Polaris) do
  defmodule NxPenalties.Integration.Polaris do
    @moduledoc """
    Gradient transformations for use with Polaris optimizers.

    These transforms operate on gradients and parameters, not on
    the loss function. They follow Polaris's composable pattern.

    ## Gradient-Level vs Loss-Level Regularization

    There are two ways to apply weight decay:

    1. **Loss-based** (NxPenalties default): Add L2 penalty to loss
       - `loss_total = loss + λ * ||w||²`
       - Gradient: `∂loss/∂w + 2λw`

    2. **Gradient transform** (Polaris style): Modify gradients directly
       - `grad_new = grad + λw`
       - Equivalent to AdamW-style decoupled weight decay

    This module provides gradient transforms for Polaris-style regularization.

    ## Composition

    Polaris transforms compose via piping:

        optimizer =
          Polaris.Optimizers.adam(learning_rate: 0.001)
          |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)
          |> NxPenalties.Integration.Polaris.add_l1_decay(0.001)

    ## Weight Decay vs L2 Regularization

    These are mathematically equivalent for SGD but differ for
    adaptive optimizers like Adam. Weight decay (implemented here)
    is generally preferred for modern training.

    Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
    """

    @doc """
    Add L2 weight decay to gradients.

    This is equivalent to Polaris's built-in weight decay in AdamW,
    provided for completeness and explicit composition.

    Weight decay modifies the gradient: g' = g + λw
    Where λ is the decay rate and w is the weight.

    ## Parameters

      * `optimizer` - Polaris optimizer tuple `{init_fn, update_fn}`
      * `decay` - Decay rate. Default: `0.01`

    ## Note

    For AdamW, prefer using the built-in `:decay` option:

        Polaris.Optimizers.adamw(learning_rate: 0.001, decay: 0.01)

    This transform is useful when you want to add decay to an optimizer
    that doesn't have it built-in, or for explicit composition.

    ## Example

        optimizer =
          Polaris.Optimizers.adam(learning_rate: 0.001)
          |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)
    """
    @spec add_l2_decay(term(), float()) :: term()
    def add_l2_decay(optimizer, decay \\ 0.01) do
      {base_init, base_update} = normalize_to_transform(optimizer)

      init_fn = fn params ->
        base_state = base_init.(params)
        %{base: base_state, l2_decay: decay}
      end

      update_fn = fn gradients, state, params ->
        # Add decay to gradients: g' = g + λw
        decayed_grads =
          deep_map(gradients, params, fn g, w ->
            Nx.add(g, Nx.multiply(w, state.l2_decay))
          end)

        # Apply base optimizer
        {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

        new_state = %{state | base: new_base_state}
        {updates, new_state}
      end

      {init_fn, update_fn}
    end

    @doc """
    Add L1 weight decay (sign decay) to gradients.

    Modifies the gradient: g' = g + λ * sign(w)
    This encourages sparsity in the weights.

    ## Parameters

      * `optimizer` - Polaris optimizer tuple
      * `decay` - Decay rate. Default: `0.001`

    ## Note

    L1 decay can cause weights to oscillate around zero. Consider
    using a small threshold to zero out very small weights.

    ## Example

        optimizer =
          Polaris.Optimizers.sgd(learning_rate: 0.01)
          |> NxPenalties.Integration.Polaris.add_l1_decay(0.001)
    """
    @spec add_l1_decay(term(), float()) :: term()
    def add_l1_decay(optimizer, decay \\ 0.001) do
      {base_init, base_update} = normalize_to_transform(optimizer)

      init_fn = fn params ->
        base_state = base_init.(params)
        %{base: base_state, l1_decay: decay}
      end

      update_fn = fn gradients, state, params ->
        # Add L1 decay: g' = g + λ * sign(w)
        decayed_grads =
          deep_map(gradients, params, fn g, w ->
            Nx.add(g, Nx.multiply(Nx.sign(w), state.l1_decay))
          end)

        {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

        new_state = %{state | base: new_base_state}
        {updates, new_state}
      end

      {init_fn, update_fn}
    end

    @doc """
    Add elastic net (L1 + L2) weight decay to gradients.

    Combines L1 and L2 decay:
    g' = g + λ * (α * sign(w) + (1-α) * w)

    ## Parameters

      * `optimizer` - Polaris optimizer tuple
      * `decay` - Overall decay rate. Default: `0.01`
      * `l1_ratio` - Ratio of L1 to L2 (α). Default: `0.5`

    ## Example

        optimizer =
          Polaris.Optimizers.adam(learning_rate: 0.001)
          |> NxPenalties.Integration.Polaris.add_elastic_net_decay(0.01, 0.3)
    """
    @spec add_elastic_net_decay(term(), float(), float()) :: term()
    def add_elastic_net_decay(optimizer, decay \\ 0.01, l1_ratio \\ 0.5) do
      {base_init, base_update} = normalize_to_transform(optimizer)

      init_fn = fn params ->
        base_state = base_init.(params)
        %{base: base_state, decay: decay, l1_ratio: l1_ratio}
      end

      update_fn = fn gradients, state, params ->
        alpha = state.l1_ratio
        lambda = state.decay

        decayed_grads =
          deep_map(gradients, params, fn g, w ->
            l1_term = Nx.multiply(Nx.sign(w), alpha)
            l2_term = Nx.multiply(w, 1.0 - alpha)
            decay_term = Nx.multiply(Nx.add(l1_term, l2_term), lambda)
            Nx.add(g, decay_term)
          end)

        {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

        new_state = %{state | base: new_base_state}
        {updates, new_state}
      end

      {init_fn, update_fn}
    end

    @doc """
    Add gradient norm clipping.

    Clips gradients to have a maximum global L2 norm. This is a form of
    regularization that prevents exploding gradients.

    ## Parameters

      * `optimizer` - Polaris optimizer tuple
      * `max_norm` - Maximum gradient norm. Default: `1.0`

    ## Example

        optimizer =
          Polaris.Optimizers.adam(learning_rate: 0.001)
          |> NxPenalties.Integration.Polaris.add_gradient_clipping(1.0)
    """
    @spec add_gradient_clipping(term(), float()) :: term()
    def add_gradient_clipping(optimizer, max_norm \\ 1.0) do
      {base_init, base_update} = normalize_to_transform(optimizer)

      init_fn = fn params ->
        base_state = base_init.(params)
        %{base: base_state, max_norm: max_norm}
      end

      update_fn = fn gradients, state, params ->
        # Compute global gradient norm
        grad_norm = compute_global_norm(gradients)

        # Compute clip factor: min(max_norm / norm, 1.0)
        clip_factor = Nx.min(Nx.divide(state.max_norm, Nx.max(grad_norm, 1.0e-8)), 1.0)

        # Scale all gradients by clip factor
        clipped_grads =
          deep_map_single(gradients, fn g ->
            Nx.multiply(g, clip_factor)
          end)

        {updates, new_base_state} = base_update.(clipped_grads, state.base, params)

        new_state = %{state | base: new_base_state}
        {updates, new_state}
      end

      {init_fn, update_fn}
    end

    @doc """
    Add Gaussian noise to gradients.

    A form of regularization that can help escape local minima and
    improve generalization. Implements a decaying noise schedule.

    ## Parameters

      * `optimizer` - Polaris optimizer tuple
      * `variance` - Base noise variance. Default: `0.01`
      * `opts` - Options:
        * `:decay` - Variance decay rate per step. Default: `0.55`
        * `:seed` - Random seed for reproducible noise. Default: random

    ## Noise Schedule

    Noise variance at step t: σ²(t) = variance / (1 + t)^decay

    Reference: "Adding Gradient Noise Improves Learning for Very Deep Networks"
    (Neelakantan et al., 2015)

    ## Example

        optimizer =
          Polaris.Optimizers.sgd(learning_rate: 0.01)
          |> NxPenalties.Integration.Polaris.add_gradient_noise(0.01, decay: 0.55)
    """
    @spec add_gradient_noise(term(), float(), keyword()) :: term()
    def add_gradient_noise(optimizer, variance \\ 0.01, opts \\ []) do
      decay = Keyword.get(opts, :decay, 0.55)
      seed = Keyword.get(opts, :seed)
      {base_init, base_update} = normalize_to_transform(optimizer)

      init_fn = fn params ->
        base_state = base_init.(params)

        key =
          if seed,
            do: Nx.Random.key(seed),
            else: Nx.Random.key(System.unique_integer([:positive]))

        %{
          base: base_state,
          variance: variance,
          decay: decay,
          step: 0,
          key: key
        }
      end

      update_fn = fn gradients, state, params ->
        # Compute current variance with decay: σ² = η / (1 + t)^γ
        current_var = state.variance / :math.pow(1 + state.step, state.decay)
        current_std = :math.sqrt(current_var)

        # Add noise to each gradient tensor
        {noisy_grads, new_key} = add_noise_to_gradients(gradients, current_std, state.key)

        {updates, new_base_state} = base_update.(noisy_grads, state.base, params)

        new_state = %{state | base: new_base_state, step: state.step + 1, key: new_key}

        {updates, new_state}
      end

      {init_fn, update_fn}
    end

    @doc """
    Add adaptive gradient clipping (AGC).

    Clips gradients based on the ratio of gradient norm to parameter norm,
    which is more stable than absolute clipping for varying parameter scales.

    ## Parameters

      * `optimizer` - Polaris optimizer tuple
      * `clip_factor` - Maximum allowed gradient-to-parameter ratio. Default: `0.01`
      * `opts` - Options:
        * `:eps` - Small constant for numerical stability. Default: `1.0e-3`

    ## Algorithm

    For each parameter w with gradient g:
    - Compute unit-wise norm ratio: ||g|| / max(||w||, eps)
    - If ratio > clip_factor: scale g by clip_factor * ||w|| / ||g||

    Reference: "High-Performance Large-Scale Image Recognition Without Normalization"
    (Brock et al., 2021)

    ## Example

        optimizer =
          Polaris.Optimizers.adam(learning_rate: 0.001)
          |> NxPenalties.Integration.Polaris.add_adaptive_gradient_clipping(0.01)
    """
    @spec add_adaptive_gradient_clipping(term(), float(), keyword()) :: term()
    def add_adaptive_gradient_clipping(optimizer, clip_factor \\ 0.01, opts \\ []) do
      eps = Keyword.get(opts, :eps, 1.0e-3)
      {base_init, base_update} = normalize_to_transform(optimizer)

      init_fn = fn params ->
        base_state = base_init.(params)
        %{base: base_state, clip_factor: clip_factor, eps: eps}
      end

      update_fn = fn gradients, state, params ->
        clipped_grads =
          deep_map(gradients, params, fn g, w ->
            # Compute norms
            g_norm = Nx.sqrt(Nx.sum(Nx.pow(g, 2)))
            w_norm = Nx.sqrt(Nx.sum(Nx.pow(w, 2)))

            # Max ratio allowed
            max_g_norm = Nx.multiply(Nx.max(w_norm, state.eps), state.clip_factor)

            # Clip if gradient norm exceeds threshold
            scale = Nx.min(Nx.divide(max_g_norm, Nx.max(g_norm, 1.0e-8)), 1.0)
            Nx.multiply(g, scale)
          end)

        {updates, new_base_state} = base_update.(clipped_grads, state.base, params)

        new_state = %{state | base: new_base_state}
        {updates, new_state}
      end

      {init_fn, update_fn}
    end

    @doc """
    Add gradient centralization.

    Centralizes gradients by subtracting their mean, which can improve
    training stability and convergence.

    ## Parameters

      * `optimizer` - Polaris optimizer tuple
      * `opts` - Options:
        * `:axes` - Axes to centralize over. Default: all except first (batch)

    Reference: "Gradient Centralization: A New Optimization Technique for DNNs"
    (Yong et al., 2020)

    ## Example

        optimizer =
          Polaris.Optimizers.adam(learning_rate: 0.001)
          |> NxPenalties.Integration.Polaris.add_gradient_centralization()
    """
    @spec add_gradient_centralization(term(), keyword()) :: term()
    def add_gradient_centralization(optimizer, opts \\ []) do
      {base_init, base_update} = normalize_to_transform(optimizer)

      init_fn = fn params ->
        base_state = base_init.(params)
        %{base: base_state, opts: opts}
      end

      update_fn = fn gradients, state, params ->
        centralized_grads =
          deep_map_single(gradients, fn g ->
            # Only centralize if tensor has more than 1 dimension
            if Nx.rank(g) > 1 do
              axes =
                case Keyword.get(state.opts, :axes) do
                  nil -> Enum.to_list(1..(Nx.rank(g) - 1))
                  custom_axes -> custom_axes
                end

              if axes == [] do
                g
              else
                mean = Nx.mean(g, axes: axes, keep_axes: true)
                Nx.subtract(g, mean)
              end
            else
              g
            end
          end)

        {updates, new_base_state} = base_update.(centralized_grads, state.base, params)

        new_state = %{state | base: new_base_state}
        {updates, new_state}
      end

      {init_fn, update_fn}
    end

    # ============================================================================
    # Helper Functions
    # ============================================================================

    # Deep map over two nested structures (gradients and params)
    # Check for tensors first since Nx.Tensor is a struct (which is also a map)
    defp deep_map(%Nx.Tensor{} = gradient, %Nx.Tensor{} = param, fun) do
      fun.(gradient, param)
    end

    defp deep_map(gradients, params, fun) when is_map(gradients) and is_map(params) do
      Map.new(gradients, fn {key, g} ->
        p = Map.fetch!(params, key)
        {key, deep_map(g, p, fun)}
      end)
    end

    defp deep_map(gradient, param, fun) do
      fun.(gradient, param)
    end

    # Deep map over single nested structure
    defp deep_map_single(%Nx.Tensor{} = tensor, fun) do
      fun.(tensor)
    end

    defp deep_map_single(gradients, fun) when is_map(gradients) do
      Map.new(gradients, fn {key, g} ->
        {key, deep_map_single(g, fun)}
      end)
    end

    defp deep_map_single(tensor, fun) do
      fun.(tensor)
    end

    # Compute global L2 norm of nested gradient structure
    defp compute_global_norm(gradients) do
      gradients
      |> flatten_params()
      |> Enum.map(&Nx.sum(Nx.pow(&1, 2)))
      |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
      |> Nx.sqrt()
    end

    # Add Gaussian noise to gradients
    defp add_noise_to_gradients(%Nx.Tensor{} = gradient, std, key) do
      {noise, new_key} =
        Nx.Random.normal(key, 0.0, std, shape: Nx.shape(gradient), type: Nx.type(gradient))

      {Nx.add(gradient, noise), new_key}
    end

    defp add_noise_to_gradients(gradients, std, key) when is_map(gradients) do
      {noisy_grads, final_key} =
        Enum.reduce(gradients, {%{}, key}, fn {name, g}, {acc, current_key} ->
          {noisy_g, new_key} = add_noise_to_gradients(g, std, current_key)
          {Map.put(acc, name, noisy_g), new_key}
        end)

      {noisy_grads, final_key}
    end

    # Flatten nested params to list
    defp flatten_params(%Nx.Tensor{} = tensor), do: [Nx.flatten(tensor)]

    defp flatten_params(params) when is_map(params) do
      Enum.flat_map(params, fn {_key, v} -> flatten_params(v) end)
    end

    defp flatten_params(_), do: []

    # Normalize optimizer or transform to {init, update} tuple
    defp normalize_to_transform({init, update}) when is_function(init) and is_function(update) do
      {init, update}
    end

    defp normalize_to_transform(optimizer) when is_tuple(optimizer) do
      optimizer
    end
  end
else
  defmodule NxPenalties.Integration.Polaris do
    @moduledoc """
    Polaris integration is unavailable because Polaris is not installed.
    Add `{:polaris, \"~> 0.1\"}` to your dependencies to enable these helpers.
    """

    @spec add_l2_decay(term(), float()) :: no_return()
    def add_l2_decay(_, _ \\ 0.01), do: raise_missing!()

    @spec add_l1_decay(term(), float()) :: no_return()
    def add_l1_decay(_, _ \\ 0.001), do: raise_missing!()

    @spec add_elastic_net_decay(term(), float(), float()) :: no_return()
    def add_elastic_net_decay(_, _ \\ 0.01, _ \\ 0.5), do: raise_missing!()

    @spec add_gradient_clipping(term(), float()) :: no_return()
    def add_gradient_clipping(_, _ \\ 1.0), do: raise_missing!()

    @spec add_gradient_noise(term(), float(), keyword()) :: no_return()
    def add_gradient_noise(_, _ \\ 0.01, _ \\ []), do: raise_missing!()

    @spec add_adaptive_gradient_clipping(term(), float(), keyword()) :: no_return()
    def add_adaptive_gradient_clipping(_, _ \\ 0.01, _ \\ []), do: raise_missing!()

    @spec add_gradient_centralization(term(), keyword()) :: no_return()
    def add_gradient_centralization(_, _ \\ []), do: raise_missing!()

    defp raise_missing! do
      raise ArgumentError,
            "Polaris integration is not available. Add {:polaris, \"~> 0.1\"} to your dependencies to enable NxPenalties.Integration.Polaris."
    end
  end
end
