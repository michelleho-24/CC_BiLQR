using Parameters
using LinearAlgebra
using Distributions
include(joinpath(@__DIR__, "..", "..", "BiLQR", "cc_ilqr_types.jl"))

@with_kw mutable struct CCCartpoleMDP <: CCiLQGPOMDP{AbstractVector, AbstractVector, AbstractVector}
    # cost
    Q::Matrix{Float64} = 1e-4 * I(4)
    R::Matrix{Float64} = 1e-4 * I(1)
    Q_N::Matrix{Float64} = Diagonal([1e-4, 1e-4, 1e-4, 0.1])
    Λ::Matrix{Float64} = Diagonal(fill(1e-4, 16))  # 4^2 - penalize state uncertainty

    # start and goal
    Σ0::Matrix{Float64} = diagm([1e-4, 1e-4, 1e-4, 1e-4])
    b0::MvNormal = MvNormal([0.0, π/2, 0.0, 0.0], Σ0)
    s_init::Vector{Float64} = begin
        s = rand(b0)
        s
    end
    s_goal::Vector{Float64} = [zeros(4)..., vec(zeros(4, 4))...]  # state goal + zero covariance

    # mechanics (mp is a fixed parameter, not part of state)
    δt::Float64 = 0.1
    mc::Float64 = 1.0
    mp::Float64 = 2.0
    g::Float64 = 9.81
    l::Float64 = 1.0

    # noise covariance matrices (4-state system)
    W_state_process::Matrix{Float64} = diagm([0.5, 0.1, 0.5, 0.1])
    W_process::Matrix{Float64} = diagm([0.5, 0.1, 0.5, 0.1])
    W_obs::Matrix{Float64} = 0.01 * I(4)
    W_obs_ekf::Matrix{Float64} = 0.01 * I(4)

    # chance constraint parameters
    δ::Float64 = 0.05  # risk threshold (5% allowed violation probability)
    x_max::Float64 = 4.8  # cart position constraint
    θ_max::Float64 = π  # pendulum angle constraint
end

"""Continuous-time dynamics (mean) discretized with simple Euler step.
State: [x, θ, dx, dθ]
Action: scalar horizontal force on cart
mp is read from the problem struct (fixed), not part of the state.
"""
function dyn_mean(p::CCCartpoleMDP, s::AbstractVector, a::AbstractVector)
    x, θ, dx, dθ = s
    mp = p.mp
    sinθ, cosθ = sin(θ), cos(θ)
    h = p.mc + mp * (sinθ^2)
    ds = [
        dx,
        dθ,
        (mp * sinθ * (p.l * (dθ^2) + p.g * cosθ) + a[1]) / h,
        -((p.mc + mp) * p.g * sinθ + mp * p.l * (dθ^2) * sinθ * cosθ + a[1] * cosθ) / (h * p.l)
    ]

    return s + p.δt * ds
end

dyn_noise(p::CCCartpoleMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::CCCartpoleMDP, sp::AbstractVector) = sp
obs_noise(p::CCCartpoleMDP, sp::AbstractVector) = p.W_obs
num_states(p::CCCartpoleMDP) = 4
num_actions(p::CCCartpoleMDP) = 1
num_observations(p::CCCartpoleMDP) = 4
num_sysvars(p::CCCartpoleMDP) = 0

"""Return the Cartesian positions of the cart and the pendulum tip."""
function visualize(p::CCCartpoleMDP, s::AbstractVector)
    x = s[1]
    θ = s[2]
    cart_pos = [x, 0.0]
    tip_pos = [x + p.l * sin(θ), p.l * cos(θ)]
    return (cart_pos, tip_pos)
end

function isvalidstate(p::CCCartpoleMDP, s::AbstractVector)
    x, θ, dx, dθ = s
    return -p.x_max <= x <= p.x_max && -p.θ_max <= θ <= p.θ_max
end

## Chance constraint implementation

"""
Constraint function: g(s) ≤ 0 for feasibility
Returns: [g1, g2, g3, g4] where:
  g1 = x - x_max ≤ 0
  g2 = -x - x_max ≤ 0  (i.e., x ≥ -x_max)
  g3 = θ - θ_max ≤ 0
  g4 = -θ - θ_max ≤ 0  (i.e., θ ≥ -θ_max)
"""
function constraint_func(p::CCCartpoleMDP, s::AbstractVector)
    x = s[1]
    θ = s[2]
    return [
        x - p.x_max,      # x ≤ x_max
        -x - p.x_max,     # x ≥ -x_max
        θ - p.θ_max,      # θ ≤ θ_max
        -θ - p.θ_max      # θ ≥ -θ_max
    ]
end

"""
Jacobian of constraint function: ∂g/∂s
Each row is the gradient of one constraint.
"""
function constraint_jacobian(p::CCCartpoleMDP, s::AbstractVector)
    return [
        1.0  0.0  0.0  0.0;   # ∂g1/∂s
        -1.0  0.0  0.0  0.0;  # ∂g2/∂s
        0.0  1.0  0.0  0.0;   # ∂g3/∂s
        0.0 -1.0  0.0  0.0    # ∂g4/∂s
    ]
end

num_constraints(p::CCCartpoleMDP) = 4

