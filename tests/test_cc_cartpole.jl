"""
Test file for chance-constrained Cartpole POMDP
"""

using Random
using LinearAlgebra
using Distributions

# Include the CC Cartpole problem
include(joinpath(@__DIR__, "..", "problems", "CC_Cartpole", "cc_cartpole.jl"))

function test_cc_cartpole_creation()
    println("\n=== Testing CC Cartpole POMDP Creation ===")
    
    # Create the POMDP
    pomdp = CCCartpoleMDP()
    
    # Test basic properties
    println("Number of states: ", num_states(pomdp))
    println("Number of actions: ", num_actions(pomdp))
    println("Number of observations: ", num_observations(pomdp))
    println("Number of constraints: ", num_constraints(pomdp))
    println("Risk threshold δ: ", pomdp.δ)
    
    # Test initial state
    println("\nInitial state: ", pomdp.s_init)
    println("Goal state (with covariance): ", pomdp.s_goal)
    
    # Test dynamics
    s_test = [0.0, π/4, 0.0, 0.0]
    a_test = [1.0]
    s_next = dyn_mean(pomdp, s_test, a_test)
    println("\nTest dynamics:")
    println("  State: ", s_test)
    println("  Action: ", a_test)
    println("  Next state: ", s_next)
    
    # Test process noise
    W = dyn_noise(pomdp, s_test, a_test)
    println("\nProcess noise covariance: ", size(W))
    
    # Test observation model
    z = obs_mean(pomdp, s_next)
    V = obs_noise(pomdp, s_next)
    println("\nObservation:")
    println("  Observation mean: ", z)
    println("  Observation noise covariance: ", size(V))
    
    # Test constraints
    println("\n=== Testing Constraints ===")
    
    # Test feasible state
    s_feasible = [2.0, 0.5, 0.0, 0.0]
    g_feasible = constraint_func(pomdp, s_feasible)
    println("\nFeasible state: ", s_feasible)
    println("Constraint values g(s): ", g_feasible)
    println("All constraints satisfied? ", all(g_feasible .<= 0))
    
    # Test state near boundary
    s_boundary = [4.5, 0.0, 0.0, 0.0]
    g_boundary = constraint_func(pomdp, s_boundary)
    println("\nBoundary state: ", s_boundary)
    println("Constraint values g(s): ", g_boundary)
    println("All constraints satisfied? ", all(g_boundary .<= 0))
    
    # Test infeasible state
    s_infeasible = [5.0, 0.0, 0.0, 0.0]
    g_infeasible = constraint_func(pomdp, s_infeasible)
    println("\nInfeasible state: ", s_infeasible)
    println("Constraint values g(s): ", g_infeasible)
    println("All constraints satisfied? ", all(g_infeasible .<= 0))
    
    # Test constraint Jacobian
    C = constraint_jacobian(pomdp, s_test)
    println("\nConstraint Jacobian at ", s_test, ":")
    println(C)
    println("Size: ", size(C))
    
    # Test initial belief construction
    println("\n=== Testing Initial Belief ===")
    b0 = vcat(pomdp.s_init, vec(pomdp.Σ0))
    println("Initial belief size: ", length(b0))
    println("State mean: ", b0[1:4])
    println("Covariance (flattened): ", b0[5:end])
    
    println("\n✓ CC Cartpole POMDP tests completed successfully!")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_cc_cartpole_creation()
end
