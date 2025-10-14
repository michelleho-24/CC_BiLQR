"""
Test file for chance-constrained LightDark POMDP
"""

using Random
using LinearAlgebra
using Distributions

# Include the LightDark problem
include(joinpath(@__DIR__, "..", "problems", "LightDark", "LightDark.jl"))

function test_lightdark_creation()
    println("\n=== Testing LightDark POMDP Creation ===")
    
    # Create the POMDP
    pomdp = LightDarkPOMDP()
    
    # Test basic properties
    println("Number of states: ", num_states(pomdp))
    println("Number of actions: ", num_actions(pomdp))
    println("Number of observations: ", num_observations(pomdp))
    println("Number of constraints: ", num_constraints(pomdp))
    println("Risk threshold δ: ", pomdp.δ)
    
    # Test initial state
    println("\nInitial state: ", pomdp.s_init)
    println("Goal state (with variance): ", pomdp.s_goal)
    println("Light location: ", pomdp.light_loc)
    
    # Test dynamics
    s_test = [5.0]  # Start at y = 5
    a_test = [-2.0]  # Move left by 2
    s_next = dyn_mean(pomdp, s_test, a_test)
    println("\nTest dynamics:")
    println("  State: ", s_test)
    println("  Action: ", a_test)
    println("  Next state: ", s_next)
    
    # Test process noise
    W = dyn_noise(pomdp, s_test, a_test)
    println("\nProcess noise covariance: ", W)
    
    # Test observation model - state-dependent noise
    println("\n=== Testing State-Dependent Observation Noise ===")
    
    # Near light source (low noise)
    s_near_light = [10.0]
    z_near = obs_mean(pomdp, s_near_light)
    V_near = obs_noise(pomdp, s_near_light)
    println("\nNear light (y = 10):")
    println("  Observation mean: ", z_near)
    println("  Observation variance: ", V_near[1, 1])
    
    # Far from light source (high noise)
    s_far_light = [50.0]
    z_far = obs_mean(pomdp, s_far_light)
    V_far = obs_noise(pomdp, s_far_light)
    println("\nFar from light (y = 50):")
    println("  Observation mean: ", z_far)
    println("  Observation variance: ", V_far[1, 1])
    
    # At goal (also far from light)
    s_goal_test = [0.0]
    z_goal = obs_mean(pomdp, s_goal_test)
    V_goal = obs_noise(pomdp, s_goal_test)
    println("\nAt goal (y = 0):")
    println("  Observation mean: ", z_goal)
    println("  Observation variance: ", V_goal[1, 1])
    
    # Test constraints
    println("\n=== Testing Constraints ===")
    
    # Test feasible state
    s_feasible = [20.0]
    g_feasible = constraint_func(pomdp, s_feasible)
    println("\nFeasible state: ", s_feasible)
    println("Constraint values g(s): ", g_feasible)
    println("All constraints satisfied? ", all(g_feasible .<= 0))
    
    # Test state near boundary
    s_boundary = [95.0]
    g_boundary = constraint_func(pomdp, s_boundary)
    println("\nBoundary state: ", s_boundary)
    println("Constraint values g(s): ", g_boundary)
    println("All constraints satisfied? ", all(g_boundary .<= 0))
    
    # Test infeasible state
    s_infeasible = [105.0]
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
    println("State mean: ", b0[1])
    println("Variance: ", b0[2])
    
    # Test clamping at boundaries
    println("\n=== Testing Boundary Clamping ===")
    s_extreme = [95.0]
    a_extreme = [10.0]  # Try to go beyond boundary
    s_clamped = dyn_mean(pomdp, s_extreme, a_extreme)
    println("From state ", s_extreme, " with action ", a_extreme)
    println("Result (should be clamped to 100): ", s_clamped)
    
    println("\n✓ LightDark POMDP tests completed successfully!")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_lightdark_creation()
end
