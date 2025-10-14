using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include(joinpath(@__DIR__, "cartpole.jl"))
include(joinpath(@__DIR__, "..", "..", "BiLQR", "bilqr.jl"))
include(joinpath(@__DIR__, "..", "..", "BiLQR", "ekf.jl"))

global b, s_true

function system_control(seed, method)
    Random.seed!(seed)

    # Initialize the Cartpole MDP
    pomdp = CartpoleMDP()

    # true initial state
    s_true = pomdp.s_init

    # initial belief: mean + covariance
    b = vcat(s_true, vec(pomdp.Σ0))

    println("b", size(b))
    println("s_true", size(s_true))

    # Simulation parameters
    num_steps = 50

    all_s = []
    all_b = []
    all_u = []
    info_dict = Dict()

    for t in 1:num_steps
        push!(all_s, s_true)
        push!(all_b, b)

        if method == "bilqr"
            results = bilqr(pomdp, b)
            if results === nothing
                return nothing
            else
                a, info_dict = results
            end
        else
            a = [0.0]
            info_dict = Dict()
        end

        push!(all_u, a)

        # Simulate the true next state
        s_next_true = dyn_mean(pomdp, s_true, a)

        # Add process noise to the true state
        noise_state = rand(MvNormal(pomdp.W_state_process))
        s_next_true = s_next_true + noise_state

        # Generate observation from the true next state
        z = obs_mean(pomdp, s_next_true)

        # Add observation noise
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), pomdp.W_obs))
        z = z + obsnoise

        # Use EKF to update the belief
        b = ekf(pomdp, b, a, z)
        if b === nothing
            return nothing
        end

        # Update the true state for the next iteration
        s_true = s_next_true
    end

    return all_b, all_s, all_u, info_dict
end
