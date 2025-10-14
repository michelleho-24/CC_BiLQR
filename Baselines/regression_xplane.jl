using LinearAlgebra
using Distributions
using ForwardDiff

function flatten_regression(X, Y, prior_mean, prior_covariance)
    try
        # Dimensions based on A (4,4) and B (4,2)
        d_x, d_u, d_out = 4, 2, 4  # Dimensions of x, u, and x_{t+1} respectively
        d_params = d_x + d_u       # Total params per input vector (6)
        n = size(X, 1)             # Number of examples

        # Reshape Y into a flat vector Y'
        Y_prime = reshape(Y', d_out * n)

        # Initialize a sparse matrix X' of dimensions (n * 4, 24)
        X_prime = spzeros(n * d_out, d_params * d_out)

        # Fill X' with block-diagonal entries
        for i in 1:n
            x_u_vec = X[i, :]  # (6,) vector for each time step
            start_idx = (i - 1) * d_out + 1
            end_idx = i * d_out
            X_prime[start_idx:end_idx, :] = kron(I(d_out), x_u_vec')
        end

        # Convert X' to dense for Bayesian update calculations
        X_dense = Matrix(X_prime)

        # Known columns for A and B as provided
        col2 = [0.0, -0.1, -0.5, 0.0]
        col3 = [-9.81, 1.0, -0.1, 1.0]
        col4 = [0.0, 0.0, 0.0, 0.0]
        colB = [0.0, 0.0, 0.0, 0.0]

        # Construct expanded prior mean for A and B (size 24)
        expanded_prior_mean = vcat(
            # First column of A (uncertain)
            prior_mean[1:4],
            # Known columns 2, 3, 4 of A
            col2, col3, col4,
            # First column of B (uncertain)
            prior_mean[5:8],
            # Known column 2 of B
            colB
        )

        # Construct expanded prior covariance matrix
        expanded_prior_covariance = Matrix{Float64}(I, 24, 24) * 1e-3  # Small values for known elements
        expanded_prior_covariance[1:8, 1:8] = prior_covariance  # Set prior covariance for unknown columns

        # Bayesian posterior update for all parameters
        Σ_0_inv = inv(expanded_prior_covariance)
        Sigma_posterior = inv(Σ_0_inv + X_dense' * X_dense)
        mu_posterior = Sigma_posterior * (Σ_0_inv * expanded_prior_mean + X_dense' * Y_prime)

        # Reshape posterior mean to match full (6, 4) structure for Θ
        Θ = reshape(mu_posterior, d_params, d_out)

        # Transpose back to get A and B in their original forms
        A_estimate = Θ[1:d_x, :]'  # Transpose to get A in (4, 4)
        B_estimate = Θ[d_x+1:end, :]'  # Transpose to get B in (4, 2)

        # Extract the relevant covariance for first columns of A and B
        A_indices = 1:4  # First 4 elements in Θ_prime correspond to A's first column
        B_indices = 5:8  # Next 4 elements correspond to B's first column

        # Retrieve only the diagonal elements for the covariance of the first columns
        Cov_A_B_first_col_diag = diagm(vcat(diag(Sigma_posterior)[A_indices], diag(Sigma_posterior)[B_indices]))

        return vcat(A_estimate[:, 1], B_estimate[:, 1]), Cov_A_B_first_col_diag

    catch e
        if isa(e, SingularException) || isa(e, LinearAlgebra.LAPACKException)
            println("Skipping seed $seed due to error: $e")
            return nothing 
        end 
    end
end

function regression(pomdp, b, method)
    iters = 2
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
        x_u_vec = vcat(state[1:4], a)  # Concatenate true state and action
        push!(data_inputs, x_u_vec)    # Store as row vector for X
        push!(data_targets, o)         # Store target next state as row vector for Y

        # Update state
        state = s_new

        # Perform regression every 2 iterations
        if t % 2 == 0 && length(data_targets) > 0
            # Convert data inputs and targets to matrices
            X = hcat(data_inputs...)'
            Y = hcat(data_targets...)'

            # Flattened regression to estimate A and B
            result = flatten_regression(X, Y, b[5:12], diagm(b[end-num_sysvars(pomdp) + 1:end]))
            if result === nothing
                return nothing
            else
                AB_estimate, Cov_Θ_prime = result
            end 

            # Update belief b with A and B estimates
            μ_new = copy(b[1:12])
            # println(size(A_estimate))
            # println(size(B_estimate))
            μ_new[5:12] = AB_estimate  # Flattened A and B

            Σ_new = diagm(b[13:end])
            Σ_new[5:12, 5:12] = Cov_Θ_prime  # Update covariance for A and B

            # Pack updated mean and covariance into belief
            b_new = [μ_new; diag(Σ_new)]
            b = b_new

            # Store estimates and variances for plotting
            push!(AB_estimated_list, AB_estimate)  # Store A and B flattened
            push!(AB_variances_list, Cov_Θ_prime)
        end
    end

    ΣΘΘ = diagm(b[end-num_sysvars(pomdp) + 1:end])

    return all_b, AB_estimated_list, AB_variances_list, ΣΘΘ, all_s, all_u, pomdp.AB_true
end


### Full A and B regression
# A = (8,8)
# B = (8,3)
# n (1000) examples of x (8,), u (3,), x_t+1 (8,)

# Normal least squares
#         X         Th   =     Y
# 1:   [x1; u1 ] [ A^T ] = [ x1_t+1]
# 2:   [x2; u2 ] [ B^T ] = [ x2_t+1]
# 3:   [ ....  ]
# ///
# 1000:[...... ]

# X * Th = Y
# Th = X\Y
# X (1000, 11)
# Th = (11, 8)
# Y = (1000, 8)

# ......
# X' (8000, 88)
# Th' (88, 1)
# Y' = (8000, 1 )

# Y' - stack of all x_t+1 (make sure corresponds to how A and B get stacked as vectors)
# Th' also stacked 

# [x1_t+1(1)]
# [x1_t+1(2)]
#....
# 

# X' (careful of how stacked ..  beware if neeed transposing or 1000 later)
# (1000 of these) -  block diagonals are nonzero
# [x1u1 ---- ---- ---- ----- ---- ---- ----]
# [---- x1u1 ---- ---- ----- ---- ---- x1u1]
# [---- ---- x1u1 x1u1 x1u1u x1u1 x1u1 x1u1]
# [---- x1u1 x1u1 x1u1 x1u1u x1u1 x1u1 x1u1]
# [---- x1u1 x1u1 x1u1 x1u1u x1u1 x1u1 x1u1]
# [---- x1u1 x1u1 x1u1 x1u1u x1u1 x1u1 x1u1]
# [---- x1u1 x1u1 x1u1 x1u1u x1u1 x1u1 x1u1]
# [---- x1u1 x1u1 x1u1 x1u1u x1u1 x1u1 x1u1]
# [---- x1u1 x1u1 x1u1 x1u1u x1u1 x1u1 x1u1]
# (8, 88) (where we are inserting x1;u1 at 11*(i-1)+1:11*i)

# multiplying now gives components of x_t+1
# 8 targets for each new data point

# Then invert X'\Y' for Th' (88,1) -rearrange for A and B depending on how flattened 
# Similar inversion process for Sigma (88, 88)

# Check flatten(Th) = Th'

# flattened will help to get the uncertainty  