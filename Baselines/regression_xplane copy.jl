using LinearAlgebra
using Distributions
using ForwardDiff

function regression(pomdp, b, method)
    iters = 200
    state = copy(pomdp.s_init)

    # True A and B parameters (unknown to the estimator)
    AB_true = pomdp.AB_true  # True parameters (length 8)

    # Lists to store estimated A and B over time
    AB_estimated_list = []
    AB_variances_list = []

    # Other variables to store simulation data
    all_s = []
    all_b = []
    all_u = []

    δt = pomdp.δt
    data_inputs = []
    data_targets = []

    for t in 1:iters
        push!(all_s, state)
        push!(all_b, b)

        # Select an action
        if method == "mpcreg"
            a = mpc(pomdp, b, 10)  # Assuming mpc function exists
        else
            a = xplane_random_policy(pomdp, b)  # Replace with your random policy function
        end

        push!(all_u, a)

        # Simulate dynamics
        s_new = dyn_mean(pomdp, state, a)
        
        # Generate observation with noise
        o = obs_mean(pomdp, s_new)
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), pomdp.W_obs))
        o += obsnoise  

        # Store inputs and outputs for regression
        push!(data_inputs, (state, a))
        push!(data_targets, o)

        # Update state
        state = s_new

        # Perform regression every 2 iterations
        if t % 2 == 0 && length(data_targets) > 0
            # Build H and Y matrices with explicit Float64 type
            T = length(data_targets)
            H = Matrix{Float64}(undef, T * 4, 8)  # Assuming each Jacobian is 4x8
            Y = Vector{Float64}(undef, T * 4)

            # Adjust H and Y initialization based on (4, 8) Jacobian size
            T = length(data_targets)
            H = Matrix{Float64}(undef, T * 4, 8)  # (4, 8) Jacobian per time step for the unknown parameters
            Y = Vector{Float64}(undef, T * 4)

            for i in 1:T
                s_i, a_i = data_inputs[i]
                o_i = data_targets[i]

                # Predicted next state using only the true state part in dyn_mean
                s_next_pred = dyn_mean(pomdp, s_i, a_i)

                # Compute Jacobian with respect to the last 8 elements (unknown parameters) only
                J_dyn = Float64.(ForwardDiff.jacobian(
                    ab -> dyn_mean(pomdp, vcat(s_i[1:4], ab...), a_i)[1:4],  # Focus on true state dynamics
                    s_i[5:end]  # Only last 8 elements (unknown parameters) vary
                ))

                H[(i-1)*4+1:i*4, :] .= J_dyn

                # Observed difference (Y) based on predicted state
                obs_pred = obs_mean(pomdp, s_next_pred)  # Predicted observation based on s_next_pred
                obs_diff = Float64.(o_i - obs_pred)  # Ensure Float64 type for observation difference

                Y[(i-1)*4+1:i*4] .= obs_diff
            end


            # Least squares solution for AB_true
            AB_estimated = H \ Y

            # Calculate residuals and covariance for AB_estimated
            residuals = Y - H * AB_estimated
            σ² = sum(residuals .^ 2) / (T * 4 - length(AB_estimated))
            Cov_Θ = σ² * pinv(H' * H)

            # println(Cov_Θ)

            # Update belief b
            μ_new = copy(b[1:12])
            μ_new[5:12] = AB_estimated  # Indices of A and B in the mean vector

            Σ_new = diagm(b[13:end])
            Σ_new[5:12, 5:12] = Cov_Θ  # Update covariance for A and B

            # Pack updated mean and covariance into belief
            b_new = [μ_new; diag(Σ_new)]
            b = b_new

            # Store estimates and variances
            push!(AB_estimated_list, AB_estimated)
            push!(AB_variances_list, Cov_Θ)
        end
    end

    # Final covariance

    return all_b, AB_estimated_list, AB_variances_list, ΣΘΘ, all_s, all_u, pomdp.AB_true
end
