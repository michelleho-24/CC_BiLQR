using Parameters
using LinearAlgebra
using Distributions

include(joinpath(@__DIR__, "..", "..", "BiLQR", "cc_ilqr_types.jl"))

"""
LightDark POMDP for chance-constrained control.
State: 1D position y
The goal is to reach y = 0. Observations are noisy, with noise increasing with distance from light_loc.
"""

@with_kw mutable struct LightDarkPOMDP <: CCiLQGPOMDP{AbstractVector, AbstractVector, AbstractVector}
    
    # cost
    Q::Matrix{Float64} = Diagonal([1.0])  # state cost
    R::Matrix{Float64} = Diagonal([0.01])  # control cost
    Q_N::Matrix{Float64} = Diagonal([10.0])  # terminal state cost
    Λ::Matrix{Float64} = Diagonal([1.0])  # belief uncertainty cost (1x1 covariance)

    # start and goal
    Σ0::Matrix{Float64} = diagm([9.0])  # initial variance (std = 3)
    b0::MvNormal = MvNormal([2.0], Σ0)
    s_init::Vector{Float64} = begin
        s = rand(b0)
        s
    end
    s_goal::Vector{Float64} = [0.0, 0.0]  # goal position + zero variance

    # dynamics
    δt::Float64 = 1.0  # time step (discrete actions)
    max_y::Float64 = 100.0  # position bounds
    light_loc::Float64 = 10.0  # location of light source

    # noise covariance matrices
    W_process::Matrix{Float64} = diagm([0.01])  # small process noise
    W_obs::Matrix{Float64} = diagm([1.0])  # will be state-dependent via obs_noise
    W_obs_ekf::Matrix{Float64} = diagm([1.0])

    # chance constraint parameters
    δ::Float64 = 0.05  # risk threshold (5% allowed violation probability)
    y_max::Float64 = 100.0  # position constraint
end

"""
Dynamics: simple integrator with clamping to bounds.
State: [y]
Action: [dy] - change in position
"""
function dyn_mean(p::LightDarkPOMDP, s::AbstractVector, a::AbstractVector)
    y = s[1]
    y_next = clamp(y + a[1] * p.δt, -p.max_y, p.max_y)
    return [y_next]
end

"""
Process noise (constant)
"""
dyn_noise(p::LightDarkPOMDP, s::AbstractVector, a::AbstractVector) = p.W_process

"""
Observation model: observe position with state-dependent noise.
Noise increases with distance from light_loc.
"""
obs_mean(p::LightDarkPOMDP, sp::AbstractVector) = sp

"""
Observation noise: variance increases with distance from light source
"""
function obs_noise(p::LightDarkPOMDP, sp::AbstractVector)
    y = sp[1]
    sigma = abs(y - p.light_loc) + 1e-4
    return diagm([sigma^2])
end

num_states(p::LightDarkPOMDP) = 1
num_actions(p::LightDarkPOMDP) = 1
num_observations(p::LightDarkPOMDP) = 1
num_sysvars(p::LightDarkPOMDP) = 0

"""
Check if state is valid (within bounds)
"""
function isvalidstate(p::LightDarkPOMDP, s::AbstractVector)
    y = s[1]
    return -p.max_y <= y <= p.max_y
end

## Chance constraint implementation

"""
Constraint function: g(s) ≤ 0 for feasibility
Returns: [g1, g2] where:
  g1 = y - y_max ≤ 0
  g2 = -y - y_max ≤ 0  (i.e., y ≥ -y_max)
"""
function constraint_func(p::LightDarkPOMDP, s::AbstractVector)
    y = s[1]
    return [
        y - p.y_max,      # y ≤ y_max
        -y - p.y_max      # y ≥ -y_max
    ]
end

"""
Jacobian of constraint function: ∂g/∂s
Each row is the gradient of one constraint.
"""
function constraint_jacobian(p::LightDarkPOMDP, s::AbstractVector)
    return [
        1.0;   # ∂g1/∂y
        -1.0   # ∂g2/∂y
    ]
end

num_constraints(p::LightDarkPOMDP) = 2