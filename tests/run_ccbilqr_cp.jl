"""
Test script for chance-constrained BiLQR on CC Cartpole
"""

# https://arxiv.org/abs/2108.06533

# CC POMDP says keep under this value not keep in region for this time step

using Random
using LinearAlgebra
using Distributions
using Plots

# Include the CC Cartpole problem and controller
include(joinpath(@__DIR__, "..", "problems", "CC_Cartpole", "cc_cartpole.jl"))
include(joinpath(@__DIR__, "..", "BiLQR", "cc_bilqr.jl"))
include(joinpath(@__DIR__, "..", "BiLQR", "ekf.jl"))

function test_cc_cartpole_control(seed=1)
    Random.seed!(seed)
    
    println("\n" * "="^60)
    println("Testing CC-BiLQR on Cartpole")
    println("="^60)
    
    # Create the CC Cartpole POMDP
    pomdp = CCCartpoleMDP()
    
    println("\nProblem setup:")
    println("  States: ", num_states(pomdp))
    println("  Actions: ", num_actions(pomdp))
    println("  Constraints: ", num_constraints(pomdp))
    println("  Risk threshold δ: ", pomdp.δ)
    println("  Cart bounds: [-$(pomdp.x_max), $(pomdp.x_max)]")
    println("  Angle bounds: [-$(pomdp.θ_max), $(pomdp.θ_max)]")
    
    # Initial true state
    s_true = pomdp.s_init
    
    # Initial belief: mean + covariance
    b = vcat(s_true, vec(pomdp.Σ0))
    
    println("\nInitial state: ", s_true)
    println("Initial belief dimension: ", length(b))
    
    # Simulation parameters
    num_steps = 50
    
    all_s = []
    all_b = []
    all_u = []
    all_constraints = []
    
    println("\n" * "="^60)
    println("Running simulation...")
    println("="^60)
    
    for t in 1:num_steps
        println("\nStep $t:")
        
        push!(all_s, s_true)
        push!(all_b, b)
        
        # Compute control using CC-BiLQR
        println("  Computing control with CC-BiLQR...")
        result = cc_bilqr(pomdp, b, N=20, max_iters=50, μ_barrier_init=1.0)
        
        if result === nothing
            println("  ⚠ CC-BiLQR failed (constraint violation)")
            break  # Stop simulation if controller fails
        end
        
        a, info_dict = result
        println("  Control: ", round.(a, digits=4))
        println("  Converged: ", info_dict[:converged])
        println("  Iterations: ", info_dict[:iterations])
        println("  Cost: ", round(info_dict[:cost], digits=4))
        
        # Check constraint satisfaction
        g = constraint_func(pomdp, s_true)
        constraint_satisfied = all(g .<= 0)
        push!(all_constraints, maximum(g))
        
        println("  State: x=$(round(s_true[1], digits=3)), θ=$(round(s_true[2], digits=3)) ($(round(rad2deg(s_true[2]), digits=1))°)")
        println("  Max constraint violation: $(round(maximum(g), digits=6))")
        println("  Constraints satisfied: ", constraint_satisfied ? "✓" : "✗")
        
        # Check if pole has fallen (angle > 60 degrees from vertical)
        if abs(s_true[2]) > π/3
            println("  ⚠ WARNING: Pole is falling! θ = $(round(rad2deg(s_true[2]), digits=1))°")
        end
        
        # Check if we should stop due to failure
        if abs(s_true[2]) > π/2
            println("  ✗ FAILURE: Pole has fallen over!")
            break
        end
        
        push!(all_u, a)
        
        # Simulate true next state
        s_next_true = dyn_mean(pomdp, s_true, a)
        
        # Add process noise
        noise_state = rand(MvNormal(pomdp.W_state_process))
        s_next_true = s_next_true + noise_state
        
        # Generate observation
        z = obs_mean(pomdp, s_next_true)
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), pomdp.W_obs))
        z = z + obsnoise
        
        # Update belief using EKF
        b = ekf(pomdp, b, a, z)
        if b === nothing
            println("EKF update failed")
            return nothing
        end
        
        s_true = s_next_true
    end
    
    println("\n" * "="^60)
    println("Simulation completed!")
    println("="^60)
    
    # Plot results
    plot_results(all_s, all_u, all_constraints, pomdp)
    
    # Create animation
    animate_cartpole(all_s, pomdp)
    
    return nothing  # Don't return the huge data arrays
end

function animate_cartpole(all_s, pomdp)
    println("\nCreating animation...")
    
    anim = @animate for (i, s) in enumerate(all_s)
        cart_pos, tip_pos = visualize(pomdp, s)
        
        # Create the plot
        plot(xlim=(-6, 6), ylim=(-0.5, 2.5), 
             aspect_ratio=:equal, 
             legend=false, 
             title="Cartpole - Step $i",
             xlabel="Position (m)",
             ylabel="Height (m)")
        
        # Draw the track
        plot!([-6, 6], [0, 0], linewidth=2, color=:black)
        
        # Draw constraint bounds
        vline!([-pomdp.x_max, pomdp.x_max], linestyle=:dash, color=:red, linewidth=1.5, alpha=0.5)
        
        # Draw the cart (as a rectangle)
        cart_width = 0.4
        cart_height = 0.3
        cart_x = [cart_pos[1] - cart_width/2, cart_pos[1] + cart_width/2, 
                  cart_pos[1] + cart_width/2, cart_pos[1] - cart_width/2, 
                  cart_pos[1] - cart_width/2]
        cart_y = [0, 0, cart_height, cart_height, 0]
        plot!(cart_x, cart_y, fillrange=0, fillalpha=0.5, color=:blue, linewidth=2)
        
        # Draw the pole
        plot!([cart_pos[1], tip_pos[1]], [cart_height/2, tip_pos[2]], 
              linewidth=4, color=:red)
        
        # Draw the pole tip
        scatter!([tip_pos[1]], [tip_pos[2]], markersize=8, color=:red)
        
        # Add state information
        annotate!(-5.5, 2.2, text("x = $(round(s[1], digits=2)) m", :left, 10))
        annotate!(-5.5, 2.0, text("θ = $(round(rad2deg(s[2]), digits=1))°", :left, 10))
    end
    
    gif(anim, "cc_cartpole_animation.gif", fps=10)
    println("Animation saved to cc_cartpole_animation.gif")
end

function plot_results(all_s, all_u, all_constraints, pomdp)
    # Extract state trajectories
    x = [s[1] for s in all_s]
    θ = [s[2] for s in all_s]
    dx = [s[3] for s in all_s]
    dθ = [s[4] for s in all_s]
    
    u = [a[1] for a in all_u]
    
    t = 0:length(all_s)-1
    
    # Create plots
    p1 = plot(t, x, label="Cart position", xlabel="Time", ylabel="x [m]", linewidth=2)
    hline!(p1, [pomdp.x_max, -pomdp.x_max], label="Bounds", linestyle=:dash, color=:red)
    
    p2 = plot(t, θ, label="Pole angle", xlabel="Time", ylabel="θ [rad]", linewidth=2)
    hline!(p2, [pomdp.θ_max, -pomdp.θ_max], label="Bounds", linestyle=:dash, color=:red)
    
    p3 = plot(0:length(u)-1, u, label="Control", xlabel="Time", ylabel="Force [N]", linewidth=2, marker=:circle)
    
    p4 = plot(0:length(all_constraints)-1, all_constraints, label="Max constraint violation", 
              xlabel="Time", ylabel="g(s)", linewidth=2, marker=:circle)
    hline!(p4, [0.0], label="Feasibility", linestyle=:dash, color=:green, linewidth=2)
    
    plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800), plot_title="CC-BiLQR Cartpole Control")
    
    savefig("cc_cartpole_results.png")
    println("\nPlot saved to cc_cartpole_results.png")
end

# Run test
if abspath(PROGRAM_FILE) == @__FILE__
    test_cc_cartpole_control(1)
end
