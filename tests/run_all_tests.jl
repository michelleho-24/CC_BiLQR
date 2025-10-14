"""
Master test runner for all chance-constrained POMDP tests
"""

println("=" ^ 60)
println("Running Chance-Constrained POMDP Tests")
println("=" ^ 60)

# Test CC Cartpole
include("test_cc_cartpole.jl")
test_cc_cartpole_creation()

println("\n" * "=" ^ 60)

# Test LightDark
include("test_lightdark.jl")
test_lightdark_creation()

println("\n" * "=" ^ 60)
println("All tests completed!")
println("=" ^ 60)
