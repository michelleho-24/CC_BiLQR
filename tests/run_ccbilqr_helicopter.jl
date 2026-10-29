"""
Test script for chance-constrained BiLQR on Helicopter Navigation
"""

using Random
using LinearAlgebra
using Distributions
using Plots

# Include the Helicopter problem and controller
include(joinpath(@__DIR__, "..", "problems", "Helicopter", "helicopter.jl"))
include(joinpath(@__DIR__, "..", "BiLQR", "cc_bilqr.jl"))
include(joinpath(@__DIR__, "..", "BiLQR", "ekf.jl"))

function test_helicopter_navigation(seed=1)
    Random.seed!(seed)
    
    println("\n" * "="^60)
    println("Testing CC-BiLQR on Helicopter Navigation")
    println("="^60)
    
    # Create the Helicopter MDP
    pomdp = HelicopterMDP()
    
    println("\nProblem setup:")
    println("  States: ", num_states(pomdp))
    println("  Actions: ", num_actions(pomdp))
    println("  Constraints: ", num_constraints(pomdp))
    println("  Risk threshold δ: ", pomdp.δ)
    println("  Dangerous zones: ", length(pomdp.danger_zones))
    println("  Start position: ", pomdp.s_init[1:3])
    println("  Goal position: ", pomdp.s_goal[1:3])
    
    # Initial true state
    s_true = pomdp.s_init
    
    # Initial belief: mean + covariance
    b = vcat(s_true, vec(pomdp.Σ0))
    
    println("\nInitial state: ", s_true)
    println("Initial belief dimension: ", length(b))
    
    # Simulation parameters
    num_steps = 100
    
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
        result = cc_bilqr(pomdp, b, N=15, max_iters=50, μ_barrier_init=1.0)
        
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
        
        x, y, z = s_true[1:3]
        println("  Position: x=$(round(x, digits=2)), y=$(round(y, digits=2)), z=$(round(z, digits=2))")
        println("  Max constraint violation: $(round(maximum(g), digits=6))")
        println("  Constraints satisfied: ", constraint_satisfied ? "✓" : "✗")
        
        # Check distance to goal
        goal_dist = norm(s_true[1:3] - pomdp.s_goal[1:3])
        println("  Distance to goal: $(round(goal_dist, digits=2)) m")
        
        # Check if reached goal
        if reached_goal(pomdp, s_true, 2.0)
            println("  ✓ Reached goal!")
            push!(all_u, a)
            break
        end
        
        push!(all_u, a)
        
        # Simulate true next state
        s_next_true = dyn_mean(pomdp, s_true, a)
        
        # Add process noise (wind disturbance)
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
            break
        end
        
        s_true = s_next_true
    end
    
    println("\n" * "="^60)
    println("Simulation completed!")
    println("="^60)
    
    # Plot results
    plot_results(all_s, all_u, all_constraints, pomdp)
    
    # Create 3D animation
    animate_helicopter_3d(all_s, pomdp)
    
    return nothing  # Don't return the huge data arrays
end

function animate_helicopter_3d(all_s, pomdp)
    println("\nCreating 3D animation...")
    
    # Get trajectory bounds for dynamic plot limits
    x_all = [s[1] for s in all_s]
    y_all = [s[2] for s in all_s]
    z_all = [s[3] for s in all_s]
    
    x_min, x_max = minimum(x_all) - 10, maximum(x_all) + 10
    y_min, y_max = minimum(y_all) - 10, maximum(y_all) + 10
    z_min_plot, z_max_plot = max(0, minimum(z_all) - 2), maximum(z_all) + 5
    
    anim = @animate for (i, s) in enumerate(all_s)
        x, y, z = s[1:3]
        
        # Create 3D plot with dynamic limits
        plot(legend=:topright, 
             xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
             xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(z_min_plot, z_max_plot),
             title="Helicopter Navigation - Step $i",
             camera=(45, 30))
        
        # Plot dangerous zones as spheres
        for (idx, zone) in enumerate(pomdp.danger_zones)
            cx, cy, cz, r = zone
            
            # Draw sphere (simplified as circles at different z-slices)
            θ = range(0, 2π, length=30)
            for dz in [-r/2, 0, r/2]
                z_slice = cz + dz
                r_slice = sqrt(max(0, r^2 - dz^2))
                x_circle = cx .+ r_slice .* cos.(θ)
                y_circle = cy .+ r_slice .* sin.(θ)
                z_circle = fill(z_slice, length(θ))
                plot!(x_circle, y_circle, z_circle, 
                      color=:red, alpha=0.3, linewidth=1, label=(dz == 0 && idx == 1 ? "Danger Zones" : ""))
            end
        end
        
        # Plot trajectory so far
        if i > 1
            x_traj = [s[1] for s in all_s[1:i]]
            y_traj = [s[2] for s in all_s[1:i]]
            z_traj = [s[3] for s in all_s[1:i]]
            plot!(x_traj, y_traj, z_traj, 
                  color=:blue, linewidth=2, label="Trajectory", marker=:circle, markersize=2)
        end
        
        # Plot current position
        scatter!([x], [y], [z], 
                markersize=10, color=:green, label="Helicopter")
        
        # Plot start and goal
        scatter!([pomdp.s_init[1]], [pomdp.s_init[2]], [pomdp.s_init[3]], 
                markersize=8, color=:cyan, marker=:square, label="Start")
        scatter!([pomdp.s_goal[1]], [pomdp.s_goal[2]], [pomdp.s_goal[3]], 
                markersize=8, color=:yellow, marker=:star5, label="Goal")
        
        # Plot waypoints
        for (idx, wp) in enumerate(pomdp.waypoints)
            scatter!([wp[1]], [wp[2]], [wp[3]], 
                    markersize=6, color=:orange, marker=:diamond, 
                    label=(idx == 1 ? "Waypoints" : ""))
        end
    end
    
    gif(anim, "helicopter_animation.gif", fps=5)
    println("Animation saved to helicopter_animation.gif")
end

function plot_results(all_s, all_u, all_constraints, pomdp)
    # Extract state trajectories
    x = [s[1] for s in all_s]
    y = [s[2] for s in all_s]
    z = [s[3] for s in all_s]
    vx = [s[4] for s in all_s]
    vy = [s[5] for s in all_s]
    vz = [s[6] for s in all_s]
    
    # Extract control trajectories
    ux = [a[1] for a in all_u]
    uy = [a[2] for a in all_u]
    uz = [a[3] for a in all_u]
    
    t = 0:length(all_s)-1
    
    # Create plots
    p1 = plot(t, x, label="X position", xlabel="Time", ylabel="X (m)", linewidth=2, color=:blue)
    plot!(p1, t, y, label="Y position", linewidth=2, color=:green)
    hline!(p1, [pomdp.xy_max, -pomdp.xy_max], label="XY Bounds", linestyle=:dash, color=:red)
    
    p2 = plot(t, z, label="Altitude", xlabel="Time", ylabel="Z (m)", linewidth=2, color=:purple)
    hline!(p2, [pomdp.z_min, pomdp.z_max], label="Altitude Bounds", linestyle=:dash, color=:red)
    
    p3 = plot(t, vx, label="vx", xlabel="Time", ylabel="Velocity (m/s)", linewidth=2, color=:blue)
    plot!(p3, t, vy, label="vy", linewidth=2, color=:green)
    plot!(p3, t, vz, label="vz", linewidth=2, color=:purple)
    hline!(p3, [pomdp.v_max, -pomdp.v_max], label="Velocity Bounds", linestyle=:dash, color=:red)
    
    p4 = plot(0:length(ux)-1, ux, label="ax", xlabel="Time", ylabel="Acceleration (m/s²)", 
              linewidth=2, marker=:circle, color=:blue)
    plot!(p4, 0:length(uy)-1, uy, label="ay", linewidth=2, marker=:circle, color=:green)
    plot!(p4, 0:length(uz)-1, uz, label="az", linewidth=2, marker=:circle, color=:purple)
    hline!(p4, [pomdp.max_accel, -pomdp.max_accel], label="Accel Bounds", linestyle=:dash, color=:red)
    
    p5 = plot(0:length(all_constraints)-1, all_constraints, label="Max constraint violation", 
              xlabel="Time", ylabel="g(s)", linewidth=2, marker=:circle, color=:red)
    hline!(p5, [0.0], label="Feasibility", linestyle=:dash, color=:green, linewidth=2)
    
    # 2D trajectory plot (top-down view)
    p6 = plot(x, y, label="Trajectory", xlabel="X (m)", ylabel="Y (m)", 
              linewidth=2, color=:blue, aspect_ratio=:equal)
    scatter!(p6, [pomdp.s_init[1]], [pomdp.s_init[2]], 
            markersize=10, color=:green, marker=:square, label="Start")
    scatter!(p6, [pomdp.s_goal[1]], [pomdp.s_goal[2]], 
            markersize=10, color=:red, marker=:star5, label="Goal")
    
    # Plot danger zones (top-down view)
    for zone in pomdp.danger_zones
        cx, cy, cz, r = zone
        θ = range(0, 2π, length=50)
        x_circle = cx .+ r .* cos.(θ)
        y_circle = cy .+ r .* sin.(θ)
        plot!(p6, x_circle, y_circle, fillrange=0, fillalpha=0.3, 
              color=:red, linewidth=2, label="")
    end
    
    # Plot waypoints
    for wp in pomdp.waypoints
        scatter!(p6, [wp[1]], [wp[2]], markersize=6, color=:orange, marker=:diamond, label="")
    end
    
    plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1400, 1200), 
         plot_title="CC-BiLQR Helicopter Navigation")
    
    savefig("helicopter_results.png")
    println("\nPlot saved to helicopter_results.png")
end

# Run test
# Uncomment the line below to run automatically when included
test_helicopter_navigation(1)
