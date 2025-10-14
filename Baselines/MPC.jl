using LinearAlgebra
using ForwardDiff
using Distributions
using POMDPs
using Infiltrator
using Optim

function cost(Q, R, Q_N, s, u, s_goal)
    # Compute the cost of a state-action pair
    return (s - s_goal)' * Q * (s - s_goal) + u' * R * u + (s - s_goal)' * Q_N * (s - s_goal)
end

function mpc(pomdp::iLQGPOMDP, initial_belief::AbstractVector, horizon::Int)
    n_actions = num_actions(pomdp)
    initial_actions = [rand(n_actions) for _ in 1:horizon]
    flat_initial_actions = reduce(vcat, initial_actions)

    function cost_function(flat_actions)
        actions = [flat_actions[(i-1)*n_actions+1:i*n_actions] for i in 1:horizon]
        belief = initial_belief
        total_cost = 0.0

        for t in 1:horizon
            action = actions[t]
            state = dyn_mean(pomdp, belief[1:num_states(pomdp)], action) # + dyn_noise(pomdp, belief[1:num_states(mdp)], action)

            total_cost += cost(pomdp.Q, pomdp.R, pomdp.Q_N, state, action, pomdp.s_goal[1:num_states(pomdp)])
        end

        return total_cost
    end

    result = optimize(cost_function, flat_initial_actions, method=BFGS())
    opt_action = result.minimizer[1:n_actions]

    return opt_action
end
