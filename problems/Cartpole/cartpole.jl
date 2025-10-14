using Parameters
include(joinpath(@__DIR__, "..", "..", "BiLQR", "ilqr_types.jl"))

@with_kw mutable struct CartpoleMDP <: iLQGPOMDP{AbstractVector, AbstractVector, AbstractVector}
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
end

"""Continuous-time dynamics (mean) discretized with simple Euler step.
State: [x, θ, dx, dθ]
Action: scalar horizontal force on cart
mp is read from the problem struct (fixed), not part of the state.
"""
function dyn_mean(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)
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

dyn_noise(p::CartpoleMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::CartpoleMDP, sp::AbstractVector) = sp
obs_noise(p::CartpoleMDP, sp::AbstractVector) = p.W_obs
num_states(p::CartpoleMDP) = 4
num_actions(p::CartpoleMDP) = 1
num_observations(p::CartpoleMDP) = 4
num_sysvars(p::CartpoleMDP) = 0

"""Return the Cartesian positions of the cart and the pendulum tip."""
function visualize(p::CartpoleMDP, s::AbstractVector)
    x = s[1]
    θ = s[2]
    cart_pos = [x, 0.0]
    tip_pos = [x + p.l * sin(θ), p.l * cos(θ)]
    return (cart_pos, tip_pos)
end

function isvalidstate(p::CartpoleMDP, s::AbstractVector)
    x, θ, dx, dθ = s
    return -4.8 <= x <= 4.8 && -pi <= θ <= pi
end
