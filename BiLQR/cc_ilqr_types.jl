using POMDPs

## Chance-Constrained POMDP type

abstract type CCiLQGPOMDP{S,A,O} <: POMDP{S,A,O} end

# interface (inherits from iLQGPOMDP interface)
"""
    dyn_mean(p::CCiLQGPOMDP, s::AbstractVector, a::AbstractVector)::AbstractVector

    Return the mean dynamics update from state `s` with control `a`.
"""
function dyn_mean end

"""
    dyn_noise(p::CCiLQGPOMDP, s::AbstractVector, a::AbstractVector)::AbstractMatrix

    Return the covariance of the dynamics update from state `s` with control `a`.
"""
function dyn_noise end

"""
    obs_mean(p::CCiLQGPOMDP, sp::AbstractVector)::AbstractVector

    Return the mean observation from state `sp`.
"""
function obs_mean end

"""
    obs_noise(p::CCiLQGPOMDP, sp::AbstractVector)::AbstractMatrix

    Return the covariance of the observation from state `sp`.
"""
function obs_noise end 

"""
    num_states(p::CCiLQGPOMDP)::Int

    Return the dimensionality of the state space in the POMDP.
"""
function num_states end

"""
    num_actions(p::CCiLQGPOMDP)::Int

    Return the dimensionality of the action space in the POMDP.
"""
function num_actions end

"""
    num_observations(p::CCiLQGPOMDP)::Int

    Return the dimensionality of the observation space in the POMDP.
"""
function num_observations end

# default reward function included in POMDP definition
Q(p::CCiLQGPOMDP) = p.Q
Q_N(p::CCiLQGPOMDP) = p.Q_N
R(p::CCiLQGPOMDP) = p.R
Λ(p::CCiLQGPOMDP) = p.Λ
s_goal(p::CCiLQGPOMDP) = p.s_goal

## Chance constraint interface

"""
    constraint_func(p::CCiLQGPOMDP, s::AbstractVector)::AbstractVector

    Return the constraint function value at state `s`.
    Should return a vector where each element g_i(s) ≤ 0 represents a constraint.
"""
function constraint_func end

"""
    constraint_jacobian(p::CCiLQGPOMDP, s::AbstractVector)::AbstractMatrix

    Return the Jacobian of the constraint function at state `s`.
    Each row corresponds to ∇g_i(s).
"""
function constraint_jacobian end

"""
    risk_threshold(p::CCiLQGPOMDP)::Float64

    Return the acceptable risk threshold δ for constraint violation.
    The chance constraint ensures P(g(s) > 0) ≤ δ.
"""
risk_threshold(p::CCiLQGPOMDP) = p.δ

"""
    num_constraints(p::CCiLQGPOMDP)::Int

    Return the number of constraints in the problem.
"""
function num_constraints end

### consistent policy interface for later user

