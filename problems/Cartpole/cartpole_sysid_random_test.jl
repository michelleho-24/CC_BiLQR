using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include("../BiLQR/ekf.jl")
include("cartpole_sysid.jl")

Random.seed!(1234)

# Initialize the Cartpole MDP
pomdp = CartpoleMDP()

# True mass of the pole (unknown to the estimator)
mp_true = 2.0  # True mass of the pole

# Initial true state
s_true = pomdp.s_init  # [x, θ, dx, dθ, mp]

# Initial belief state
Σ0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.1])  # Initial covariance
b = vcat(pomdp.s_init, Σ0[:])  # Belief vector containing mean and covariance

# Simulation parameters
num_steps = 100

# Data storage for plotting
mp_estimates = zeros(num_steps)
mp_variances = zeros(num_steps)
time_steps = collect(1:num_steps)

for t in 1:num_steps
    global s_true, b

    # Randomly sample an action (force between -10 and 10)
    a = [rand() * 20.0 - 10.0]
    
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
    
    # Extract the mean and covariance from belief
    m = b[1:num_states(pomdp)]
    Σ = reshape(b[num_states(pomdp) + 1:end], num_states(pomdp), num_states(pomdp))
    
    # Store estimates
    mp_estimates[t] = m[5]
    mp_variances[t] = Σ[5, 5]
    
    # Update the true state for the next iteration
    s_true = s_next_true
end

ΣΘΘ = b[end]

# RMSE
RMSE = sqrt(1/(num_states(mdp)*(num_states(mdp) + num_actions(mdp)))*tr(ΣΘΘ))

println("Trace of Covariance: ", tr(ΣΘΘ))
println("RMSE: ", RMSE)

# # Plot the estimated mass of the pole over time
# plot(time_steps, mp_estimates, ribbon=sqrt.(mp_variances), label="Estimated mp ± 1 std dev", xlabel="Time Step", ylabel="Estimated mp", title="EKF Estimation of mp")
# hline!([mp_true], label="True mp", linestyle=:dash)

# # Show the plot
# display(plot!)
