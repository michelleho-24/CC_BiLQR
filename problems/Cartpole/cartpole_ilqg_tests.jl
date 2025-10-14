
include("../BiLQR/ilqr_types.jl")
include("cartpole.jl")
include("../BiLQR/bilqr.jl")
include("../BiLQR/ekf.jl")

using Plots

mdp = CartpoleMDP()

b0 = [mdp.s_init..., vec(1e-3 * Matrix{Float64}(I, num_states(mdp), num_states(mdp)))...]
iters = 10
b = b0
s_bar = zeros(num_states(mdp), iters)

for i in 1:iters
    global b
    a, info_dict = iLQG(mdp, b)
    println("action ", a)
    println("cost ", info_dict[:cost])
    s_bar = info_dict[:s_bar]

    # get first state after action 
    s_bar = info_dict[:s_bar]
    s = s_bar[:, 2]

    # get observation
    z = obs_mean(mdp, s) + obs_noise(mdp, s)

    # update belief 
    b = ekf(mdp, b, a, z)

end 

# Initialize arrays to store coordinates and time
cart_x_coords = []
cart_y_coords = []
counterweight_x_coords = []
counterweight_y_coords = []
time_coords = []

# Collect coordinates from each column of s_bar
for i in 1:size(s_bar, 2)
    cart_coords, counterweight_coords = visualize(mdp, s_bar[:,i])
    println(cart_coords, counterweight_coords)
    cart_x, cart_y = cart_coords
    counterweight_x, counterweight_y = counterweight_coords
    push!(cart_x_coords, cart_x)
    push!(cart_y_coords, cart_y)
    push!(counterweight_x_coords, counterweight_x)
    push!(counterweight_y_coords, counterweight_y)
    push!(time_coords, i)  # Assuming each column corresponds to a time step
end

# Function to create a rectangle shape centered at (x, y)
function centered_rectangle_shape(x, y, width, height)
    return Shape([x - width/2, x + width/2, x + width/2, x - width/2], [y - height/2, y - height/2, y + height/2, y + height/2])
end

# Create the animation
animation = @animate for i in 1:length(time_coords)
    println("Processing frame $i")
    plot(
        centered_rectangle_shape(cart_x_coords[i], cart_y_coords[i], 0.01, 0.1), seriestype=:shape, label="Cart", 
        xlabel="X", ylabel="Y", title="Cart-pole Random example", 
        legend=:topright, fillalpha=0.5, color=:blue
    )
    plot!([counterweight_x_coords[i]], [counterweight_y_coords[i]], seriestype=:scatter, label="Counterweight", color=:red)
    plot!([cart_x_coords[i], counterweight_x_coords[i]], [cart_y_coords[i], counterweight_y_coords[i]], seriestype=:path, color=:black, label=false)
end

# Save the animation as a GIF
println("Saving animation as GIF")
gif(animation, "plots/random_cart_counterweight_animation.gif", fps=2)
println("Animation saved successfully")


# println(mp_cov_per_timestep)

# # Plot cost vs time
# plot(1:iters, cost_per_timestep, xlabel = "Time", ylabel = "Cost", title = "$(method): Cost vs time")
# savefig("$(method)_cost_vs_time.png")

# # Plot cumulative cost vs time
# plot(1:iters, cumsum(cost_per_timestep), xlabel = "Time", ylabel = "Cumulative Cost", title = "$(method): Cumulative Cost vs time")
# savefig("$(method)_cumulative_cost_vs_time.png")

# # plot variance of belief of mass of pole vs time
# plot(1:iters, mp_cov_per_timestep, xlabel = "Time", ylabel = "Variance of mass of pole", title = "$(method): Variance of mass of pole vs time")
# savefig("$(method)_mp_cov_vs_time.png")