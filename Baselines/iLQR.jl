
using LinearAlgebra
using ForwardDiff
using Distributions
using POMDPs
using Infiltrator
using Optim

# Define the iLQR algorithm

function iLQR(pomdp, x0::AbstractVector, U::AbstractMatrix, N::Int, max_iter::Int, tol::Float64)
    n_controls = size(U, 1)
    n_states = length(x0)
    
    # Initialize
    X = zeros(n_states, N + 1)
    X[:, 1] = x0
    L = zeros(n_controls, n_states, N)  # Feedback gain
    u_update = zeros(n_controls, N)
    V = 0.0

    for iter in 1:max_iter
        # Forward pass to compute nominal trajectory
        for k in 1:N
            X[:, k+1] = dyn_mean(pomdp, X[:, k], U[:, k])
        end
        
        # Backward pass: compute cost-to-go and feedback gains
        Vxx = zeros(n_states, n_states)
        Vx = zeros(n_states)

        for k in N:-1:1
            # Compute cost and its derivatives (quadratic approximation)
            lx, lu, lxx, luu, lux = cost_derivatives(pomdp, X[:, k], U[:, k])
            fx = ForwardDiff.jacobian(x -> dyn_mean(pomdp, x, U[:, k]), X[:, k])
            fu = ForwardDiff.jacobian(u -> dyn_mean(pomdp, X[:, k], u), U[:, k])

            # Backward recursion for V
            Qx = lx + fx' * Vx
            Qu = lu + fu' * Vx
            Qxx = lxx + fx' * Vxx * fx
            Quu = luu + fu' * Vxx * fu
            Qux = lux + fu' * Vxx * fx

            # Compute feedback gain and update controls
            L[:, :, k] = -inv(Quu) * Qux
            u_update[:, k] = -inv(Quu) * Qu

            Vx = Qx + L[:, :, k]' * Quu * u_update[:, k]
            Vxx = Qxx + L[:, :, k]' * Quu * L[:, :, k]
        end

        # Line search and control update
        alpha = 1.0
        X_new = copy(X)
        for k in 1:N
            U[:, k] += alpha * u_update[:, k]
            X_new[:, k+1] = dyn_mean(pomdp, X_new[:, k], U[:, k])
        end

        # Check convergence
        if norm(X_new - X) < tol
            println("Converged after $iter iterations")
            break
        end
        
        X = X_new
    end

    return X, U
end

# Cost function derivatives
function cost_derivatives(pomdp, x::AbstractVector, u::AbstractVector)
    l = cost(pomdp, x, u)
    lx = ForwardDiff.gradient(x -> cost(pomdp, x, u), x)
    lu = ForwardDiff.gradient(u -> cost(pomdp, x, u), u)
    lxx = ForwardDiff.hessian(x -> cost(pomdp, x, u), x)
    luu = ForwardDiff.hessian(u -> cost(pomdp, x, u), u)
    lux = ForwardDiff.jacobian(u -> ForwardDiff.gradient(x -> cost(pomdp, x, u), x), u)

    return lx, lu, lxx, luu, lux
end

# Belief update for iLQR
function update_belief(pomdp::iLQGPOMDP, belief::AbstractVector, u::AbstractVector)
    n_states = num_states(pomdp)
    x = belief[1:n_states]
    Σ = reshape(belief[n_states+1:end], n_states, n_states)

    x_new = dyn_mean(pomdp, x, u)
    Σ_pred = sigma_pred(pomdp, x, u, Σ)
    Σ_new = sigma_update(pomdp, x_new, Σ_pred)
    
    belief_new = vcat(x_new, vec(Σ_new))
    return belief_new
end

# Sigma prediction function
function sigma_pred(pomdp, x::AbstractVector, u::AbstractVector, Σ::AbstractMatrix)
    At = ForwardDiff.jacobian(x -> dyn_mean(pomdp, x, u), x)
    return At * Σ * At' + dyn_noise(pomdp, x,u)
end

# Sigma update function
function sigma_update(pomdp, x::AbstractVector, Σ::AbstractMatrix)
    Ct = ForwardDiff.jacobian(x -> obs_mean(pomdp,x), x)
    K = Σ * Ct' * inv(Ct * Σ * Ct' + obs_noise(pomdp, x))
    return (I - K * Ct) * Σ
end

