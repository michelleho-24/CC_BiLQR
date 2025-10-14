using Parameters
using LinearAlgebra
using Distributions
using Random
using Statistics  # For variance calculation
using Plots
# include("../Cartpole/cartpole_sysid.jl")
include("../Cartpole/cartpole_sysid.jl")

function regression(seed)
    Random.seed!(seed)

    # Initialize variables
    pomdp = CartpoleMDP()

    Σ0 = 0.01 * Matrix{Float64}(I, num_states(pomdp), num_states(pomdp))
    # Increase uncertainty for mp
    Σ0[end, end] = 0.1
    b0 = [pomdp.s_init...; vec(Σ0)...]
    b = b0

    iters = 200
    state = copy(pomdp.s_init)

    # True mass of the pole (unknown to the estimator)
    mp_true = 2.0  # True mass

    # Data storage for regression
    data_features = []
    data_targets = []

    # List to store estimated mass values over time
    mp_estimated_list = []
    variance_mp_list = []
    all_s = []

    # Loop over iterations
    for t in 1:iters
        # global state, b, mp_estimated 
        # Select an action (e.g., random force between -10 and 10)
        a = [rand() * 20.0 - 10.0]  # Random action between -10 and 10

        # Simulate the true next state (unknown to the estimator)
        s_true = copy(state)
        s_true[5] = mp_true  # Use true mp for simulation
        s_next_true = dyn_mean(pomdp, s_true, a)
        noise_state = rand(MvNormal(pomdp.W_state_process))
        noise_total = vcat(noise_state, 0.0)
        s_next_true += noise_total

        # Generate observation
        o = obs_mean(pomdp, s_next_true)
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), pomdp.W_obs))
        o += obsnoise

        # Collect data for regression
        x, θ, dx, dθ = state[1:4]
        x_next, θ_next, dx_next, dθ_next = o

        # Compute observed acceleration
        x_acc_obs = (dx_next - dx) / pomdp.δt

        sinθ = sin(θ)
        cosθ = cos(θ)
        dθ2 = dθ^2
        total_mass = pomdp.mc + mp_true  # Using true mp for calculation

        # Simplified linear model for x_acc_obs ≈ c1 * mp + c0
        c1 = (a[1] * (-1) - pomdp.l * dθ2 * sinθ * (dx_next - dx)) / ((pomdp.mc + mp_true) * pomdp.δt)
        c0 = (dx_next - dx) / pomdp.δt - a[1] / (pomdp.mc + mp_true)

        # Collect the data for regression
        push!(data_features, [sinθ * dθ2 / total_mass])
        push!(data_targets, x_acc_obs - a[1] / total_mass)

        # Perform the regression after every iteration
        if t % 2 == 0 && length(data_targets) > 0
            # Convert data to matrices
            X = hcat(data_features...)
            X = reshape(X, :, 1)
            y = data_targets

            # Perform linear regression
            β_est = (X' * X) \ (X' * y)
            mp_estimated = β_est[1]

            # Estimate variance of mp
            residuals = y .- X * β_est
            σ² = sum(residuals.^2) / (length(y) - 1)
            variance_mp = σ² * inv(X' * X)[1, 1]

            # Update belief b
            μ_new = copy(b[1:5])
            μ_new[5] = mp_estimated
            Σ_new = reshape(b[6:end], 5, 5)
            Σ_new[5, 5] = variance_mp

            # Pack the updated mean and covariance back into belief
            b_new = [μ_new; vec(Σ_new)]
            b = b_new

            # Store estimated mass for plotting
            push!(mp_estimated_list, mp_estimated)
            push!(variance_mp_list, variance_mp)

            # println("Iteration $t: Estimated mp = $mp_estimated, Variance of mp = $variance_mp")
        end

        # Update state estimate for next iteration
        state = s_next_true
        state[5] = b[5]  # Use current estimate of mp
        state[2] = mod(state[2] + π, 2π) - π  # Keep θ within [-π, π]

        push!(all_s, state)

    end
    # ΣΘΘ = reshape(b[end-24:end], 5, 5)
    ΣΘΘ = b[end]
    return b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s
end 

# # Final RMSE calculation
# rmse = sqrt(mean((mp_true - mp_estimated)^2))
# println("Final Estimated mp = $mp_estimated")
# println("Final RMSE= $rmse")
# # Plot the estimated mass over iterations
# # plot(1:100, mp_estimated_list, label="Estimated Pole Mass", xlabel="Iteration", ylabel="Mass of Pole", legend=:topright)
# plot(1:100, mp_estimated_list, ribbon=sqrt.(variance_mp_list), label="Estimated mp ± 1 std dev", xlabel="Time Step", ylabel="Estimated mp", title="EKF Estimation of mp")
# plot!(1:100, fill(mp_true, 100), label="True Mass", linestyle=:dash, linewidth=2, color=:red)

# # Save the plot
# savefig("mp_est_cartpole_sysid_reg_test.png")