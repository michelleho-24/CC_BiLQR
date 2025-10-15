# Inspired from https://github.com/TobiaMarcucci/gcspy/blob/main/examples/shortest_path/shortest_path.py

using JuMP
import Pajarito, HiGHS, Hypatia
using LinearAlgebra
using GraphsOfConvexSets
using Plots

model = Model(() -> Optimizer(
    optimizer_with_attributes(
        Pajarito.Optimizer,
        "oa_solver" => optimizer_with_attributes(
            HiGHS.Optimizer,
            MOI.Silent() => false,
            "mip_feasibility_tolerance" => 1e-8,
            "mip_rel_gap" => 1e-6,
        ),
        "conic_solver" =>
            optimizer_with_attributes(Hypatia.Optimizer, MOI.Silent() => false),
    )
))

@variable(model, x[1:5, 1:2])
MOI.set.(model, VariableVertexOrEdge(), x, 1:5)

# Centers
C = [
     1    0
    10    0
     4    2
     5.5 -2
     7    2
]

# source vertex
D = Diagonal([1, 1/2]) # scaling matrix
con_ref = @constraint(model, [1; D * (x[1, :] - C[1, :])] in SecondOrderCone())
MOI.set(model, ConstraintVertexOrEdge(), con_ref, 1)

# target vertex
D = Diagonal([1/2, 1]) # scaling matrix
con_ref = @constraint(model, [1; D * (x[2, :] - C[2, :])] in SecondOrderCone())
MOI.set(model, ConstraintVertexOrEdge(), con_ref, 2)
con_ref = @constraint(model, x[2, 1] <= C[2, 1]) # cut right half of the set
MOI.set(model, ConstraintVertexOrEdge(), con_ref, 2)

# vertex 1
con_ref = @constraint(model, [1; x[3, :] - C[3, :]] in MOI.NormInfinityCone(3))
MOI.set(model, ConstraintVertexOrEdge(), con_ref, 3)

# vertex 2
con_ref = @constraint(model, [1.2; x[4, :] - C[4, :]] in MOI.NormOneCone(3))
MOI.set(model, ConstraintVertexOrEdge(), con_ref, 4)
con_ref = @constraint(model, [1; x[4, :] - C[4, :]] in SecondOrderCone())
MOI.set(model, ConstraintVertexOrEdge(), con_ref, 4)

# vertex 3
con_ref = @constraint(model, [1; x[5, :] - C[5, :]] in SecondOrderCone())
MOI.set(model, ConstraintVertexOrEdge(), con_ref, 5)

edge_list = [(1, 3), (1, 4), (3, 4), (3, 5), (4, 5), (4, 2), (5, 2)]

for (src, dst) in edge_list
    # Cost of the edge
    cost = @variable(model)
    MOI.set(model, VariableVertexOrEdge(), cost, (src, dst))
    set_vertex_or_edge_objective(model, (src, dst), cost)

    cons_ref = @constraint(model, [cost; x[dst, :] - x[src, :]] in SecondOrderCone())
    MOI.set(model, ConstraintVertexOrEdge(), cons_ref, (src, dst))

    # Constraint on the edge
    cons_ref = @constraint(model, x[src, 2] <= x[dst, 2])
    MOI.set(model, ConstraintVertexOrEdge(), cons_ref, (src, dst))
end

MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
MOI.set(model, Problem(), ShortestPathProblem(1, 2))

optimize!(model)

sol = MOI.get(model, GraphsOfConvexSets.SubGraph())

using GraphPlot
gplot(sol, nodelabel = 1:5, layout = shell_layout)

xv = value.(x)

p = plot(aspect_ratio=:equal, legend=:outertopright, xlabel="x", ylabel="y", title="Shortest Path through Convex Sets")

# Helper function to draw ellipse
function draw_ellipse!(p, center, a, b; kwargs...)
    θ = range(0, 2π, length=100)
    x_ellipse = center[1] .+ a .* cos.(θ)
    y_ellipse = center[2] .+ b .* sin.(θ)
    plot!(p, x_ellipse, y_ellipse; kwargs...)
end

# Helper function to draw rectangle (infinity norm ball)
function draw_rectangle!(p, center, radius; kwargs...)
    x_rect = [center[1]-radius, center[1]+radius, center[1]+radius, center[1]-radius, center[1]-radius]
    y_rect = [center[2]-radius, center[2]-radius, center[2]+radius, center[2]+radius, center[2]-radius]
    plot!(p, x_rect, y_rect; kwargs...)
end

# Helper function to draw diamond (1-norm ball)
function draw_diamond!(p, center, radius; kwargs...)
    x_diamond = [center[1], center[1]+radius, center[1], center[1]-radius, center[1]]
    y_diamond = [center[2]+radius, center[2], center[2]-radius, center[2], center[2]+radius]
    plot!(p, x_diamond, y_diamond; kwargs...)
end

# Draw the convex sets
# Vertex 1: Ellipse with scaling [1, 1/2]
draw_ellipse!(p, C[1, :], 1.0, 0.5, linecolor=:blue, linestyle=:dash, linewidth=2, fillalpha=0.1, fillcolor=:blue, label="V1: Ellipse")

# Vertex 2: Half-ellipse with scaling [1/2, 1], cut at x <= 10
θ_half = range(π/2, 3π/2, length=50)
x_half = C[2, 1] .+ 0.5 .* cos.(θ_half)
y_half = C[2, 2] .+ 1.0 .* sin.(θ_half)
plot!(p, [x_half; C[2, 1]], [y_half; C[2, 2]+1], linecolor=:red, linestyle=:dash, linewidth=2, fillalpha=0.1, fillcolor=:red, label="V2: Half-Ellipse")

# Vertex 3: Infinity norm ball (square)
draw_rectangle!(p, C[3, :], 1.0, linecolor=:green, linestyle=:dash, linewidth=2, fillalpha=0.1, fillcolor=:green, label="V3: Square")

# Vertex 4: Intersection of 1-norm (diamond) and 2-norm (circle)
draw_diamond!(p, C[4, :], 1.2, linecolor=:orange, linestyle=:dot, linewidth=2, fillalpha=0, label="V4: Diamond")
draw_ellipse!(p, C[4, :], 1.0, 1.0, linecolor=:orange, linestyle=:dash, linewidth=2, fillalpha=0.1, fillcolor=:orange, label="V4: Circle")

# Vertex 5: Circle
draw_ellipse!(p, C[5, :], 1.0, 1.0, linecolor=:purple, linestyle=:dash, linewidth=2, fillalpha=0.1, fillcolor=:purple, label="V5: Circle")

using Graphs
# Draw the optimal path
for edge in edge_list
    if Graphs.has_edge(sol, edge)
        v = [edge...]
        plot!(p, xv[v, 1], xv[v, 2], arrow=true, label="", color=:black, linewidth=3)
    end
end

# Draw centers
scatter!(p, C[:, 1], C[:, 2], label="Centers", markerstrokewidth=0, markersize=5, color=:gray)

# Draw path points on the convex sets
for v in filter(x -> Graphs.degree(sol,x) > 0, Graphs.vertices(sol))
    plot!(p, [C[v, 1], xv[v, 1]], [C[v, 2], xv[v, 2]], label="", linestyle=:dot, color=:black, linewidth=1)
    scatter!(p, [xv[v, 1]], [xv[v, 2]], label="", markerstrokewidth=2, markersize=8, color=:red)
end

display(p)
