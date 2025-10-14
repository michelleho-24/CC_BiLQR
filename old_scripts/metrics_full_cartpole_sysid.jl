using Statistics 
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings

function log_prob_gaussian(x, mean, variance)
    return -0.5 * log(2 * π * variance) - 0.5 * ((x - mean)^2 / variance)
end

# Define the number of time steps
t = 50
plotting_seed = 1
time_steps = collect(1:t)

@load "regression_cartpolefull_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

# # Convert the ValueIterator to an array and filter out NaN values from all_ΣΘΘ
# filtered_ΣΘΘ = filter(x -> !any(isnan, x), collect(values(all_ΣΘΘ)))

# Calculate trace statistics
trace_values = [ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]
trace_regression_avg = mean(trace_values)
trace_regression_std = std(trace_values)
trace_regression_ste = trace_regression_std / sqrt(length(trace_values))

# log_probs_regression = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot(time_steps, log_probs_regression, label="Linear Regression", title = "System Identification")

last_log_probs_regression = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_regression= mean(last_log_probs_regression)
std_last_log_prob_regression = std(last_log_probs_regression)

# plot(time_steps, [(all_mp_estimates[plotting_seed][i]) for i in 1:t], ribbon=[(all_mp_variances[plotting_seed][i]) for i in 1:t], label="Regression", xlabel="Time Step", ylabel="Mass Estimate")

@load "random_cartpolefull_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_ste = trace_random_std/sqrt(length(all_ΣΘΘ))

log_probs_random = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot!(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

last_log_probs_random = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_random = mean(last_log_probs_random)
std_last_log_prob_random = std(last_log_probs_random)

plot!(time_steps, [(all_mp_estimates[plotting_seed][i]) for i in 1:t], ribbon=[(all_mp_variances[plotting_seed][i]) for i in 1:t], label="EKF", xlabel="Time Step", ylabel="Mass Estimate")

@load "bilqr_cartpolefull_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std/sqrt(length(all_ΣΘΘ))

log_probs_bilqr = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot!(time_steps, log_probs_bilqr, label="BiLQR")

last_log_probs_bilqr = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_bilqr = mean(last_log_probs_bilqr)
std_last_log_prob_bilqr = std(last_log_probs_bilqr)

plot!(time_steps, [(all_mp_estimates[plotting_seed][i]) for i in 1:t], ribbon=[(all_mp_variances[plotting_seed][i]) for i in 1:t], label="BiLQR", xlabel="Time Step", ylabel="Mass Estimate")


# # Remove the 17th element from last_log_probs_bilqr
# filtered_last_log_probs_bilqr = [last_log_probs_bilqr[i] for i in 1:length(last_log_probs_bilqr) if i != 17]

# # Calculate the average and standard deviation without the 17th element
# avg_last_log_prob_bilqr = mean(filtered_last_log_probs_bilqr)
# std_last_log_prob_bilqr = std(filtered_last_log_probs_bilqr)

# # Find seeds with high log probabilities
# sorted_indices = sortperm(filtered_last_log_probs_bilqr, rev=true)  # Sort in descending order
# high_log_prob_seeds = collect(keys(all_mp_estimates))[sorted_indices[1:5]]  # Top 5 seeds

# println("Seeds with high log probabilities: ", high_log_prob_seeds)


println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_ste)
println("Random: ", trace_random_avg, " ± ", trace_random_ste)
println("Regression: ", trace_regression_avg, " ± ", trace_regression_ste)

println("Final Log Probability")
println("BILQR: ", avg_last_log_prob_bilqr, " ± ", std_last_log_prob_bilqr)
println("Random: ", avg_last_log_prob_random, " ± ", std_last_log_prob_random)
println("Regression: ", avg_last_log_prob_regression, " ± ", std_last_log_prob_regression)

hline!([all_mp_true[plotting_seed ]], label="True mass", linestyle=:dash)

# Set default DPI and save the plot
default(dpi=1000)
savefig("est_full_mp.png")