"""
Chance-Constrained BiLQR

Based on bilqr.jl but with log barrier method for chance constraints.
"""

using LinearAlgebra
using ForwardDiff
using Distributions
using POMDPs
using Infiltrator
using SparseArrays

form_belief_vector(x,Σ) = [x..., vec(Σ)...]

function sigma_pred(pomdp, x::AbstractVector, u::AbstractVector, Σ::AbstractMatrix)
    At = ForwardDiff.jacobian(x -> dyn_mean(pomdp, x, u), x)
    return At * Σ * At' + dyn_noise(pomdp, x,u)
end

function sigma_update(pomdp, x::AbstractVector, Σ::AbstractMatrix)
    Ct = ForwardDiff.jacobian(x -> obs_mean(pomdp,x), x)
    S = Ct * Σ * Ct' + obs_noise(pomdp, x)
    if any(isnan, S) || abs(det(S)) < 1e-12
        println("CC-BiLQR S is nan, next seed...")
        return nothing
    end

    K = Σ * Ct' * inv(S)
    return (I - K * Ct) * Σ
end

function update_belief(pomdp::CCiLQGPOMDP, belief::AbstractVector, u::AbstractVector)
    # extract mean and covariance from belief state
    n_states = num_states(pomdp)
    x = belief[1:n_states]
    Σ = reshape(belief[n_states+1:end], n_states, n_states)

    # mean update belief using most likely observation (no measurement gain)
    x_new = dyn_mean(pomdp, x, u)
    Σ_pred = sigma_pred(pomdp, x,u,Σ)
    Σ_new = sigma_update(pomdp, x_new, Σ_pred)

    if Σ_new === nothing
        return nothing
    end

    return form_belief_vector(x_new,Σ_new)
end

function safe_update_belief(pomdp, belief, u)
    result = update_belief(pomdp, belief, u)
    if result === nothing
        return zeros(length(belief))
    end
    return result
end

function superAB(pomdp, q, r, N, s_bar, u_bar)
    A = zeros(N, q, q)
    B = zeros(N, q, r)

    for k in 1:N
        test_belief = update_belief(pomdp, s_bar[k, :], u_bar[k, :])
        if test_belief === nothing
            return nothing
        end
        
        try
            A[k, :, :] = ForwardDiff.jacobian(bel -> safe_update_belief(pomdp, bel, u_bar[k, :]), s_bar[k, :])
            B[k, :, :] = ForwardDiff.jacobian(u -> safe_update_belief(pomdp, s_bar[k, :], u), u_bar[k, :])
        catch e
            return nothing
        end
    end

    return A, B
end

"""
Linearize chance constraints using first-order approximation.
Returns constraint Jacobian C and offset d such that: C * b + d <= 0
"""
function linearize_constraints(pomdp::CCiLQGPOMDP, s_bar, N, δ)
    n_states = num_states(pomdp)
    nc = num_constraints(pomdp)
    q = size(s_bar, 2)  # belief dimension
    
    # Inverse normal CDF for risk threshold
    z_δ = quantile(Normal(0, 1), 1 - δ)
    
    C = zeros(N+1, nc, q)
    d = zeros(N+1, nc)
    
    for k = 1:(N+1)
        # Extract mean and covariance
        μ = s_bar[k, 1:n_states]
        Σ = reshape(s_bar[k, n_states+1:end], n_states, n_states)
        
        # Get constraint value and Jacobian
        g = constraint_func(pomdp, μ)
        ∇g = constraint_jacobian(pomdp, μ)
        
        for i = 1:nc
            # Constraint variance: ∇g[i]' * Σ * ∇g[i]
            σ_g = sqrt(max(∇g[i, :]' * Σ * ∇g[i, :], 1e-10))
            
            # Linearized chance constraint: g(μ) + z_δ * σ_g ≤ 0
            # Gradient w.r.t. mean
            C[k, i, 1:n_states] = ∇g[i, :]
            
            # Gradient w.r.t. covariance
            ∂σ_∂Σ = (∇g[i, :] * ∇g[i, :]') / (2 * σ_g)
            C[k, i, n_states+1:end] = z_δ * vec(∂σ_∂Σ)
            
            # Constraint offset
            d[k, i] = -(g[i] + z_δ * σ_g)
        end
    end
    
    return C, d
end

function cost(Q, R, Q_N, s, u, s_goal)
    return (s - s_goal)' * Q * (s - s_goal) + u' * R * u + (s - s_goal)' * Q_N * (s - s_goal)
end

# Chance-Constrained BiLQR function
function cc_bilqr(pomdp, b0; N = 10, eps=1e-3, max_iters=100, μ_barrier_init=1.0, μ_barrier_factor=0.1, μ_barrier_min=1e-6)

    if max_iters <= 1
        throw(ArgumentError("Argument `max_iters` must be at least 2."))
    end

    # define the dynamics function
    f = update_belief

    n_states = num_states(pomdp)
    num_belief_states = n_states + n_states^2
    nc = num_constraints(pomdp)
    δ = pomdp.δ  # Risk threshold

    s_goal = pomdp.s_goal

    Q = spzeros(num_belief_states, num_belief_states)
    Q[1:n_states, 1:n_states] .= pomdp.Q

    R = pomdp.R

    Q_N = spzeros(num_belief_states, num_belief_states)
    Q_N[n_states + 1:end, n_states + 1:end] .= pomdp.Λ
    Q_N[1:n_states, 1:n_states] .= pomdp.Q_N

    q = size(Q,1)  # belief dimension
    r = size(R, 1)  # control dimension

    # Initialize gains `Y` and offsets `y` for the policy
    Y = zeros(Float64, N, r, q)
    y = zeros(Float64, N, r)

    # Initialize the nominal trajectory `(s_bar, u_bar)`, and the deviations `(ds, du)`
    u_bar = zeros(Float64, N, r)
    s_bar = zeros(Float64, N+1, q)
    s_bar[1, :] = b0

    for k in 1:N
        next_belief = f(pomdp, s_bar[k, :], u_bar[k, :])
        if next_belief === nothing
            return nothing
        end
        s_bar[k+1, :] = next_belief
    end

    ds = zeros(Float64, N+1, q)
    du = zeros(Float64, N, r)

    # Barrier method: outer loop over barrier parameter
    μ_barrier = μ_barrier_init
    converged = false
    total_iters = 0

    while μ_barrier > μ_barrier_min && !converged
        println("Barrier parameter μ_barrier = $μ_barrier")
        
        # iLQR loop for current barrier parameter
        for iter in 1:max_iters
            total_iters += 1

            result = superAB(pomdp, q, r, N, s_bar, u_bar)
            if result === nothing
                return nothing
            end
            A, B = result

            # Linearize constraints
            C, d = linearize_constraints(pomdp, s_bar, N, δ)

            # Backward pass with log barrier
            V = copy(Q_N)
            v = Q_N * (s_bar[N+1, :] - s_goal)
            
            # Add log barrier for terminal constraints
            for i = 1:nc
                g_val = C[N+1, i, :]' * s_bar[N+1, :] + d[N+1, i]
                if g_val >= 0
                    # Use quadratic penalty instead of log barrier for infeasible points
                    # This allows the algorithm to start from infeasible initial trajectories
                    penalty_weight = μ_barrier * 1000.0  # Large penalty weight
                    v += penalty_weight * max(0, g_val) * C[N+1, i, :]
                    V += penalty_weight * C[N+1, i, :] * C[N+1, i, :]'
                else
                    # Use log barrier for feasible points
                    v += -μ_barrier / g_val * C[N+1, i, :]
                    V += μ_barrier / (g_val^2) * C[N+1, i, :] * C[N+1, i, :]'
                end
            end

            # Backward pass
            for k in N:-1:1
                Qxk = Q * (s_bar[k, :] - s_goal) + A[k, :, :]' * v
                Quk = R * u_bar[k, :] + B[k, :, :]' * v
                Qxx = Q + A[k, :, :]' * V * A[k, :, :]
                Quu = R + B[k, :, :]' * V * B[k, :, :]
                Qux = B[k, :, :]' * V * A[k, :, :]
                
                # Add log barrier for constraints at timestep k
                for i = 1:nc
                    g_val = C[k, i, :]' * s_bar[k, :] + d[k, i]
                    if g_val >= 0
                        # Use quadratic penalty for infeasible points
                        penalty_weight = μ_barrier * 1000.0
                        Qxk += penalty_weight * max(0, g_val) * C[k, i, :]
                        Qxx += penalty_weight * C[k, i, :] * C[k, i, :]'
                    else
                        # Use log barrier for feasible points
                        Qxk += -μ_barrier / g_val * C[k, i, :]
                        Qxx += μ_barrier / (g_val^2) * C[k, i, :] * C[k, i, :]'
                    end
                end

                y[k, :] = -inv(Quu) * Quk
                Y[k, :, :] = -inv(Quu) * Qux

                v = Qxk - Y[k, :, :]' * Quu * y[k, :]
                V = Qxx - Y[k, :, :]' * Quu * Y[k, :, :]
            end

            # Forward pass
            s_bar_prev = copy(s_bar)
            for k in 1:N
                du[k, :] = Y[k, :, :] * ds[k, :] + y[k, :]
                next_belief = f(pomdp, s_bar[k, :], u_bar[k, :] + du[k, :])
                if next_belief === nothing
                    return nothing
                end 
                s_bar[k+1, :] = next_belief
                u_bar[k, :] += du[k, :]
                ds[k+1, :] = s_bar[k+1, :] - s_bar_prev[k+1, :]
            end

            if maximum(abs.(du)) < eps
                println("  Converged at iteration $iter")
                converged = true
                break
            end
            
            if iter % 10 == 0
                println("  Iter $iter, Δu = $(maximum(abs.(du)))")
            end
        end
        
        # Reduce barrier parameter
        μ_barrier = μ_barrier * μ_barrier_factor
    end

    cost_final = cost(Q, R, Q_N, s_bar[N+1, :], u_bar[N, :], s_goal)

    info_dict = Dict(:converged => converged, :s_bar => s_bar, :u_bar => u_bar, :cost => cost_final, :iterations => total_iters)
   
    return u_bar[1,:], info_dict
end

