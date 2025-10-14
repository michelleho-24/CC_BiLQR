using JLD2
include(joinpath(@__DIR__, "..", "problems", "Cartpole", "cartpole_tests.jl"))

# Initialize dictionaries to store outputs for each seed
all_b = Dict{Int, Vector{Vector{Float64}}}()
all_s = Dict{Int, Vector{Vector{Float64}}}()
all_u = Dict{Int, Vector{Vector{Float64}}}()
all_info = Dict{Int, Any}()

method = "bilqr"

jld2_file = "cartpole_control_results.jld2"
if isfile(jld2_file)
    @load jld2_file all_b all_s all_u all_info
end

# Run the control experiment
for seed in 1:200
    println("Seed: ", seed)
    results = system_control(seed, method)
    if results === nothing
        continue
    end
    b_seed, s_seed, u_seed, info = results

    all_b[seed] = b_seed
    all_s[seed] = s_seed
    all_u[seed] = u_seed
    all_info[seed] = info

    @save jld2_file all_b all_s all_u all_info
end