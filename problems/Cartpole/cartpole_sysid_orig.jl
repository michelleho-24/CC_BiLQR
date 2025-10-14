using Parameters
include("../BiLQR/ilqr_types.jl")

@with_kw mutable struct CartpoleMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    # reward
    Q::Matrix{Float64} = 1e-6 * Matrix{Float64}(I, 5, 5)
    R::Matrix{Float64} = 1e-6 * Matrix{Float64}(I, 1, 1)
    # Q = Diagonal([1e-10, 0.1, 1e-10, 0.1, 0.1])
    # R::Matrix{Float64} = 0.05 * Matrix{Float64}(I, 1, 1)
    Q_N::Matrix{Float64} = Diagonal([1e-10, 1e-10, 1e-10, 1e-10, 0.1])
    Λ::Matrix{Float64} = Diagonal(vcat(fill(1e-10, 24), [0.1])) 
    μ0::Vector{Float64} = [0.0, π/2, 0.0, 0.0, 2.0]  # Initial mean 
    Σ0::Matrix{Float64} = Diagonal([0.5, 0.5, 0.5, 0.5, 1.0])  # Initial covariance
    b0::Vector{Float64} = vcat(μ0, Float64.(Σ0[:]))  # Initial belief vector

    s_goal::Vector{Float64} = [0.0, π/2, 0.0, 0.0, 2.0, vec(zeros(5, 5))...]  # Goal state

    # s_init::Vector{Float64} = [0.0, π/2, 0.0, 0.0, mp]
    
    # mechanics
    δt::Float64 = 0.1
    # mp::Float64 = 2.0
    mc::Float64 = 1.0
    g::Float64 = 9.81
    l::Float64 = 1.0

    # noise covariance matrices
    W_state_process::Matrix{Float64} = 1e-2 * Matrix{Float64}(I, 4, 4)
    W_process::Matrix{Float64} = Diagonal(vcat(fill(1e-2, 4), [0.000001])) 
    
    # basically fully observable - no noise in the observation
    W_obs::Matrix{Float64} = 1e-6 * Matrix{Float64}(I, 4, 4)
end

function dyn_mean(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)
    x, θ, dx, dθ, log_mp = s
    mp = exp(log_mp)
    sinθ, cosθ = sin(θ), cos(θ)
    h = p.mc + mp * (sinθ^2)
    ds = [
        dx,
        dθ,
        (mp * sinθ * (p.l * (dθ^2) + p.g * cosθ) + a[1]) / h,
        -((p.mc + mp) * p.g * sinθ + mp * p.l * (dθ^2) * sinθ * cosθ + a[1] * cosθ) / (h * p.l), 
        0.0
    ]

    # need to bound s[1] between -4.8 and 4.8, s[2] between -pi and pi
    s_new = s + p.δt * ds

    # wrap s_new[2] to be within pi of the mean which is pi/2
    # s_new[2] = π/2 + mod(s_new[2] - π/2 + π, 2π) - π
    # s_new[2] = mod(s_new[2] + π, 2π) - π
    # s_new[1] = clamp(s_new[1], -4.8, 4.8)
    # s_new[2] = clamp(s_new[2], -π, π)

    return s_new
end

dyn_noise(p::CartpoleMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::CartpoleMDP, sp::AbstractVector) = sp[1:4]
obs_noise(p::CartpoleMDP, sp::AbstractVector) = p.W_obs
num_states(p::CartpoleMDP) = 5
num_actions(p::CartpoleMDP) = 1
num_observations(p::CartpoleMDP) = 4
num_sysvars(p::CartpoleMDP) = 1

"""Return a tuple of the Cartesian positions of the cart and the counterweight"""
function visualize(p::CartpoleMDP, s::AbstractVector) 
    return ([s[1], 0.], [s[1]+p.l*sin(s[2]), p.l*cos(s[2])])
end

