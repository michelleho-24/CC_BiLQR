using Parameters
include("../BiLQR/ilqr_types.jl")

@with_kw mutable struct CartpoleMDP <: iLQGPOMDP{AbstractVector, AbstractVector, AbstractVector}
    # reward
    Q = Diagonal([1e-10, 0.1, 1e-10, 0.1, 0.1])
    R::Matrix{Float64} = 0.05 * Matrix{Float64}(I, 1, 1)
    Q_N::Matrix{Float64} = Diagonal([1e-4, 0.1, 1e-4, 0.1, 0.1])
    Λ::Matrix{Float64} = Diagonal(vcat(fill(1e-4, 24), [1]))  # 5^2
    # start and end positions
    # mp_true::Float64 = 2.0
    Σ0::Matrix{Float64} = Diagonal([1e-4, 1e-4, 1e-4, 1e-4, 2.0])
    b0::MvNormal = MvNormal([0.0, π/2, 0.0, 0.0, 2.0], Σ0)
    s_init::Vector{Float64} = begin
        s = rand(b0)
        s[end] = abs(s[end])  # Ensure the last element is positive
        s
    end
    mp_true::Float64 = s_init[end]
    s_goal::Vector{Float64} = [s_init..., vec(zeros(5, 5))...]
    # mechanics
    δt::Float64 = 0.1
    mc::Float64 = 1.0
    g::Float64 = 9.81
    l::Float64 = 1.0
    # noise covariance matrices
    W_state_process::Matrix{Float64} = Diagonal([0.5, 0.1, 0.5, 0.1])
    W_process::Matrix{Float64} = Diagonal([0.5, 0.1, 0.5, 0.1, 0.0])
    W_obs::Matrix{Float64} = 0.01 * I(2)
    W_obs_ekf::Matrix{Float64} = 0.01 * I(2)
end

function dyn_mean(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)
    x, θ, dx, dθ, mp = s
    sinθ, cosθ = sin(θ), cos(θ)
    h = p.mc + mp * (sinθ^2)
    ds = [
        dx,
        dθ,
        (mp * sinθ * (p.l * (dθ^2) + p.g * cosθ) + a[1]) / h,
        -((p.mc + mp) * p.g * sinθ + mp * p.l * (dθ^2) * sinθ * cosθ + a[1] * cosθ) / (h * p.l), 
        0.0
    ]
    
    s_new = s + p.δt * ds

    return s_new
end

dyn_noise(p::CartpoleMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::CartpoleMDP, sp::AbstractVector) = sp[1:2]
obs_noise(p::CartpoleMDP, sp::AbstractVector) = p.W_obs
num_states(p::CartpoleMDP) = 5
num_actions(p::CartpoleMDP) = 1
num_observations(p::CartpoleMDP) = 2
num_sysvars(p::CartpoleMDP) = 1

"""Return a tuple of the Cartesian positions of the cart and the counterweight"""
function visualize(p::CartpoleMDP, s::AbstractVector) 
    return ([s[1], 0.], [s[1]+p.l*sin(s[2]), p.l*cos(s[2])])
end

function isvalidstate(p::CartpoleMDP, s::AbstractVector)
    x, θ, dx, dθ, mp = s
    return -4.8 <= x <= 4.8 && -pi <= θ <= pi
end