"""
Test script for chance-constrained BiLQR on LightDark problem
"""

using Random
using LinearAlgebra
using Distributions
using Plots

# Include the LightDark problem and controller
include(joinpath(@__DIR__, "..", "problems", "LightDark", "LightDark.jl"))
include(joinpath(@__DIR__, "..", "BiLQR", "cc_bilqr.jl"))
include(joinpath(@__DIR__, "..", "BiLQR", "ekf.jl"))

function test_lightdark_navigation(seed=1)
    Random.seed!(seed)
    
    println("\n" * "="^60)
    println("Testing CC-BiLQR on LightDark Problem")
    println("="^60)
    
    # Create the LightDark POMDP
    pomdp = LightDarkPOMDP()
    
    println("\nProblem setup:")
    println("  States: ", num_states(pomdp))
    println("  Actions: ", num_actions(pomdp))
    println("  Constraints: ", num_constraints(pomdp))
    println("  Risk threshold δ: ", pomdp.δ)
    println("  Light location: ", pomdp.light_loc)
    println("  Position bounds: [-$(pomdp.y_max), $(pomdp.y_max)]")
    
    # Initial true state
    s_true = pomdp.s_init
    
    # Initial belief: mean + covariance
    b = vcat(s_true, vec(pomdp.Σ0))
    
    println("\nInitial true position: ", s_true[1])
    println("Initial belief mean: ", b[1])
    println("Initial belief variance: ", b[2])
    println("Initial belief dimension: ", length(b))
    
    # Simulation parameters
    num_steps = 30
    
    all_s = []
    all_b = []
    all_u = []
    all_constraints = []
    all_obs_noise = []
    
    println("\n" * "="^60)
    println("Running simulation...")
    println("="^60)
    
    for t in 1:num_steps
        println("\nStep $t:")
        
        push!(all_s, s_true)
        push!(all_b, b)
        
        # Get current observation noise level
        obs_noise_level = sqrt(obs_noise(pomdp, s_true)[1, 1])
        push!(all_obs_noise, obs_noise_level)
        
        # Compute control using CC-BiLQR
        println("  Computing control with CC-BiLQR...")
        result = cc_bilqr(pomdp, b, N=10, max_iters=50, μ_barrier_init=1.0)
        
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
        
        y = s_true[1]
        println("  Position: y=$(round(y, digits=3))")
        println("  Distance to light: $(round(abs(y - pomdp.light_loc), digits=3))")
        println("  Observation noise (σ): $(round(obs_noise_level, digits=3))")
        println("  Belief variance: $(round(b[2], digits=3))")
        println("  Max constraint violation: $(round(maximum(g), digits=6))")
        println("  Constraints satisfied: ", constraint_satisfied ? "✓" : "✗")
        
        # Check if reached goal
        if abs(y) < 0.5 && b[2] < 0.1  # Near goal with low uncertainty
            println("  ✓ Reached goal!")
            push!(all_u, a)
            break
        end
        
        push!(all_u, a)
        
        # Simulate true next state
        s_next_true = dyn_mean(pomdp, s_true, a)
        
        # Add process noise
        noise_state = rand(MvNormal(pomdp.W_process))
        s_next_true = s_next_true + noise_state
        
        # Generate observation
        z = obs_mean(pomdp, s_next_true)
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), obs_noise(pomdp, s_next_true)))
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
    plot_results(all_s, all_b, all_u, all_constraints, all_obs_noise, pomdp)
    
    # Create animation
    animate_lightdark(all_s, all_b, all_obs_noise, pomdp)
    
    return nothing
end

function animate_lightdark(all_s, all_b, all_obs_noise, pomdp)
    println("\nCreating animation...")
    
    # Get trajectory bounds
    y_all = [s[1] for s in all_s]
    y_min, y_max = minimum(y_all) - 5, maximum(y_all) + 5
    
    anim = @animate for (i, s) in enumerate(all_s)
        y = s[1]
        b_mean = all_b[i][1]
        b_var = all_b[i][2]
        b_std = sqrt(b_var)
        obs_std = all_obs_noise[i]
        
        # Create plot
        plot(xlim=(y_min, y_max), ylim=(0, 5),
             xlabel="Position y", ylabel="",
             legend=:topright,
             title="LightDark Problem - Step $i",
             grid=true)
        
        # Draw light source region (low observation noise)
        vspan!([pomdp.light_loc - 2, pomdp.light_loc + 2], 
               color=:yellow, alpha=0.3, label=(i==1 ? "Light Source" : ""))
        
        # Draw goal region
        vspan!([-0.5, 0.5], color=:green, alpha=0.3, label=(i==1 ? "Goal" : ""))
        
        # Draw position bounds
        vline!([pomdp.y_max, -pomdp.y_max], 
               linestyle=:dash, color=:red, linewidth=2, alpha=0.5, label=(i==1 ? "Bounds" : ""))
        
        # Draw belief distribution (mean ± 2σ)
        plot!([b_mean - 2*b_std, b_mean + 2*b_std], [2, 2],
              linewidth=10, color=:blue, alpha=0.5, label=(i==1 ? "Belief (95%)" : ""))
        scatter!([b_mean], [2], markersize=8, color=:blue, label=(i==1 ? "Belief Mean" : ""))
        
        # Draw true position
        scatter!([y], [3], markersize=12, color=:red, marker=:star5, label=(i==1 ? "True Position" : ""))
        
        # Add text annotations
        annotate!(y_min + 1, 4.5, text("True pos: $(round(y, digits=2))", :left, 10))
        annotate!(y_min + 1, 4.0, text("Belief: $(round(b_mean, digits=2)) ± $(round(b_std, digits=2))", :left, 10))
        annotate!(y_min + 1, 3.5, text("Obs noise σ: $(round(obs_std, digits=2))", :left, 10))
        
        # Mark light location
        vline!([pomdp.light_loc], linestyle=:dot, color=:orange, linewidth=2, label="")
    end
    
    gif(anim, "lightdark_animation.gif", fps=3)
    println("Animation saved to lightdark_animation.gif")
end

function plot_results(all_s, all_b, all_u, all_constraints, all_obs_noise, pomdp)
    # Extract trajectories
    y_true = [s[1] for s in all_s]
    y_belief = [b[1] for b in all_b]
    var_belief = [b[2] for b in all_b]
    std_belief = sqrt.(var_belief)
    
    u = [a[1] for a in all_u]
    
    t = 0:length(all_s)-1
    
    # Create plots
    p1 = plot(t, y_true, label="True Position", xlabel="Time", ylabel="Position y", 
              linewidth=2, color=:red, marker=:circle)
    plot!(p1, t, y_belief, label="Belief Mean", linewidth=2, color=:blue, linestyle=:dash)
    plot!(p1, t, y_belief .+ 2*std_belief, label="Belief +2σ", linewidth=1, color=:blue, linestyle=:dot, alpha=0.5)
    plot!(p1, t, y_belief .- 2*std_belief, label="Belief -2σ", linewidth=1, color=:blue, linestyle=:dot, alpha=0.5)
    hline!(p1, [0.0], label="Goal", linestyle=:dash, color=:green, linewidth=2)
    hline!(p1, [pomdp.light_loc], label="Light Location", linestyle=:dot, color=:orange, linewidth=2)
    hline!(p1, [pomdp.y_max, -pomdp.y_max], label="Bounds", linestyle=:dash, color=:red)
    
    p2 = plot(t, std_belief, label="Belief Std Dev", xlabel="Time", ylabel="σ", 
              linewidth=2, color=:purple, marker=:circle)
    
    p3 = plot(t, all_obs_noise, label="Observation Noise σ", xlabel="Time", ylabel="σ", 
              linewidth=2, color=:orange, marker=:circle)
    
    p4 = plot(0:length(u)-1, u, label="Control Input", xlabel="Time", ylabel="dy", 
              linewidth=2, marker=:circle, color=:green)
    
    p5 = plot(0:length(all_constraints)-1, all_constraints, label="Max Constraint Violation", 
              xlabel="Time", ylabel="g(s)", linewidth=2, marker=:circle, color=:red)
    hline!(p5, [0.0], label="Feasibility", linestyle=:dash, color=:green, linewidth=2)
    
    # Distance to goal over time
    dist_to_goal = abs.(y_true)
    p6 = plot(t, dist_to_goal, label="Distance to Goal", xlabel="Time", ylabel="|y|", 
              linewidth=2, color=:blue, marker=:circle)
    
    plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1400, 1200), 
         plot_title="CC-BiLQR LightDark Navigation")
    
    savefig("lightdark_results.png")
    println("\nPlot saved to lightdark_results.png")
end

# Run test
# Uncomment the line below to run automatically when included
test_lightdark_navigation(1)
