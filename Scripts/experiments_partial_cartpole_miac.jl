using JLD2
# include("../Cartpole/cartpole_sysid_tests.jl")
include("../Cartpole/cartpole_miac_tests_partial.jl")

# Initialize dictionaries to store outputs for each seed
all_b = Dict{Int, Vector{Vector{Float64}}}()
all_mp_estimates = Dict{Int, Vector{Float64}}()
all_mp_variances = Dict{Int, Vector{Float64}}()
all_ΣΘΘ = Dict{Int, Float64}()
all_s = Dict{Int, Vector{Vector{Float64}}}()
all_u = Dict{Int, Vector{Vector{Float64}}}()
all_mp_true = Dict{Int, Float64}()

method = "random"

jld2_file = "test_$(method)_cartpolepartial_miac_results.jld2"
if isfile(jld2_file)
    @load jld2_file all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true
end

# Run the system identification experiment
for seed in 1:60
    println("Seed: ", seed)
    
    results = system_identification(seed, method)
    if results === nothing
        continue
    end
    b_seed, mp_estimates_seed, mp_variances_seed, ΣΘΘ_seed, s_seed, u_seed, mp_true_seed = results

    # Store the vector with the seed as the key
    all_b[seed] = b_seed
    all_mp_estimates[seed] = mp_estimates_seed
    all_mp_variances[seed] = mp_variances_seed
    all_ΣΘΘ[seed] = ΣΘΘ_seed
    all_s[seed] = s_seed
    all_u[seed] = u_seed
    all_mp_true[seed] = mp_true_seed

    # Save all dictionaries to a JLD2 file
    @save jld2_file all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true
end