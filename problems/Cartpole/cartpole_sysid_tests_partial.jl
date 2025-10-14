# using Plots
using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include("../BiLQR/ilqr_types.jl")
include("cartpole_sysid_partial.jl")
include("../BiLQR/bilqr.jl")
include("../BiLQR/ekf.jl")
include("../Baselines/MPC.jl")
include("../Baselines/random_policy.jl")
include("../Baselines/Regression.jl")

global b, s_true

function system_identification(seed, method)
    
    Random.seed!(seed)

    # Initialize the Cartpole MDP
    pomdp = CartpoleMDP()

    # define initial distribution for total belief state 
    s_true = pomdp.s_init
    
    b = vcat(s_true[1:end - num_sysvars(pomdp)], pomdp.mp_true, pomdp.Σ0[:])
    # Simulation parameters
    num_steps = 50

    # Data storage for plotting
    mp_estimates = []
    mp_variances = []
    push!(mp_estimates, b[num_states(pomdp)])
    push!(mp_variances, b[end - 1])
    all_s = []
    all_b = []
    all_u = []

    for t in 1:num_steps

        # Store the true state for plotting
        push!(all_s, s_true)
        push!(all_b, b)

        if method == "bilqr"
            results = bilqr(pomdp, b)
            if results === nothing
                return nothing
            else 
                a, info_dict = results
            end
        elseif method == "mpc"
            a = mpc(pomdp, b, 10)
        elseif method == "random"
            a = random_policy(pomdp, b)
        elseif method == "regression" || method == "mpcreg"
            all_b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s, all_u, pomdp.mp_true = regression(pomdp, b)
            return all_b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s, all_u, pomdp.mp_true
        end 

        # a, info_dict = bilqr(pomdp, b)
        # if a == nothing
        #     break
        # end
        # a = mpc(pomdp, b, 10)
        # a = random_policy(pomdp, b)
        # all_b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s, all_u, pomdp.mp_true = regression(pomdp, b)
        # return all_b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s, all_u, pomdp.mp_true

        push!(all_u, a)
        
        # Simulate the true next state
        s_next_true = dyn_mean(pomdp, s_true, a)
        
        # Add process noise to the true state
        noise_state = rand(MvNormal(pomdp.W_state_process))
        noise_total = vcat(noise_state, 0.0)
        s_next_true = s_next_true + noise_total
        
        # Generate observation from the true next state
        z = obs_mean(pomdp, s_next_true)
        
        # Add observation noise
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), pomdp.W_obs))
        z = z + obsnoise
        
        # Use your ekf function to update the belief
        b = ekf(pomdp, b, a, z)
        if b === nothing
            return nothing
        end
        
        # Extract the mean and covariance from belief
        m = b[1:num_states(pomdp)]
        Σ = reshape(b[num_states(pomdp) + 1:end], num_states(pomdp), num_states(pomdp))
        
        # Store estimates
        # mp_estimates[t] = m[num_states(pomdp)]
        # mp_variances[t] = Σ[num_states(pomdp), num_states(pomdp)]
        push!(mp_estimates, m[num_states(pomdp)])
        push!(mp_variances, Σ[num_states(pomdp), num_states(pomdp)])
        
        # Update the true state for the next iteration
        s_true = s_next_true

    end

    ΣΘΘ = b[end]
    
    return all_b, mp_estimates, mp_variances, ΣΘΘ, all_s, all_u, pomdp.mp_true

end