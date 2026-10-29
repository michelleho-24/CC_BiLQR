"""
Ingenuity Helicopter Navigation Problem with Chance Constraints

The helicopter must navigate to waypoints and reach a landing site while avoiding
dangerous zones (high winds, obstacles, etc.) with 95% probability. The chance
constraints account for wind uncertainty affecting the helicopter's position.

State: [x, y, z, vx, vy, vz] (6D)
  - (x, y, z): 3D position
  - (vx, vy, vz): 3D velocity

Action: [ax, ay, az] (3D)
  - Acceleration commands in each direction

Chance constraints:
  - Stay out of dangerous zones with δ = 0.05 (95% confidence)
  - Dangerous zones modeled as cylinders or spheres
"""

using Parameters
using LinearAlgebra
using Distributions
include(joinpath(@__DIR__, "..", "..", "BiLQR", "cc_ilqr_types.jl"))

@with_kw mutable struct HelicopterMDP <: CCiLQGPOMDP{AbstractVector, AbstractVector, AbstractVector}
    # Cost weights
    Q::Matrix{Float64} = Diagonal([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])  # State cost (position + velocity)
    R::Matrix{Float64} = 0.1 * I(3)  # Control cost (acceleration)
    Q_N::Matrix{Float64} = Diagonal([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])  # Terminal state cost
    Λ::Matrix{Float64} = Diagonal(fill(0.1, 36))  # Covariance penalty (6^2)

    # Start and goal
    Σ0::Matrix{Float64} = diagm([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # Initial uncertainty
    s_init::Vector{Float64} = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]  # Start at (0, 0, 2m) hovering
    s_goal::Vector{Float64} = [50.0, 50.0, 1.0, 0.0, 0.0, 0.0, zeros(36)...]  # Land at (50, 50, 1m)
    
    # Waypoints (optional - can be used for multi-stage planning)
    waypoints::Vector{Vector{Float64}} = [[25.0, 15.0, 3.0], [40.0, 35.0, 2.5]]

    # Dynamics parameters
    δt::Float64 = 0.2  # Time step (200ms)
    mass::Float64 = 1.8  # kg (Ingenuity mass)
    g::Float64 = 3.71  # Mars gravity (m/s^2)
    drag_coeff::Float64 = 0.01  # Air drag coefficient (very low due to thin Mars atmosphere ~1% of Earth)
    
    # Control limits
    max_accel::Float64 = 5.0  # Maximum acceleration (m/s^2)
    
    # Noise covariance matrices (wind uncertainty)
    # Wind causes position uncertainty
    W_state_process::Matrix{Float64} = diagm([0.2, 0.2, 0.1, 0.3, 0.3, 0.15])  # Wind-induced process noise
    W_process::Matrix{Float64} = diagm([0.2, 0.2, 0.1, 0.3, 0.3, 0.15])  # For prediction
    W_obs::Matrix{Float64} = diagm([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])  # GPS/IMU observation noise
    W_obs_ekf::Matrix{Float64} = diagm([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])

    # Chance constraint parameters
    δ::Float64 = 0.05  # Risk threshold (5% allowed violation probability = 95% safety)
    
    # Dangerous zones (obstacles/high wind regions)
    # Each zone: [center_x, center_y, center_z, radius]
    danger_zones::Vector{Vector{Float64}} = [
        [15.0, 20.0, 2.0, 5.0],   # Zone 1: center at (15, 20, 2), radius 5m
        [35.0, 25.0, 3.0, 6.0],   # Zone 2: center at (35, 25, 3), radius 6m
        [45.0, 10.0, 2.5, 4.0],   # Zone 3: center at (45, 10, 2.5), radius 4m
    ]
    
    # Flight envelope constraints
    z_min::Float64 = 0.5   # Minimum altitude (m)
    z_max::Float64 = 10.0  # Maximum altitude (m)
    xy_max::Float64 = 60.0 # Horizontal boundary (m)
    v_max::Float64 = 8.0   # Maximum velocity (m/s)
end

"""
Continuous-time dynamics (mean) discretized with Euler integration.
State: [x, y, z, vx, vy, vz]
Action: [ax, ay, az] - acceleration commands

Dynamics:
  dx/dt = vx
  dy/dt = vy
  dz/dt = vz
  dvx/dt = ax - drag*vx
  dvy/dt = ay - drag*vy
  dvz/dt = az - drag*vz - g (compensate for gravity)
"""
function dyn_mean(p::HelicopterMDP, s::AbstractVector, a::AbstractVector)
    x, y, z, vx, vy, vz = s
    ax, ay, az = a
    
    # Clamp accelerations to limits
    ax = clamp(ax, -p.max_accel, p.max_accel)
    ay = clamp(ay, -p.max_accel, p.max_accel)
    az = clamp(az, -p.max_accel, p.max_accel)
    
    # Dynamics with drag and gravity
    # az is upward acceleration command, gravity pulls down
    ds = [
        vx,
        vy,
        vz,
        ax - p.drag_coeff * vx,
        ay - p.drag_coeff * vy,
        az - p.drag_coeff * vz - p.g  # Need az ≈ g to hover
    ]
    
    return s + p.δt * ds
end

dyn_noise(p::HelicopterMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::HelicopterMDP, sp::AbstractVector) = sp  # Full state observation
obs_noise(p::HelicopterMDP, sp::AbstractVector) = p.W_obs
num_states(p::HelicopterMDP) = 6
num_actions(p::HelicopterMDP) = 3
num_observations(p::HelicopterMDP) = 6
num_sysvars(p::HelicopterMDP) = 0

"""
Check if state is valid (within flight envelope).
"""
function isvalidstate(p::HelicopterMDP, s::AbstractVector)
    x, y, z, vx, vy, vz = s
    
    # Check altitude bounds
    if z < p.z_min || z > p.z_max
        return false
    end
    
    # Check horizontal bounds
    if abs(x) > p.xy_max || abs(y) > p.xy_max
        return false
    end
    
    # Check velocity bounds
    v = sqrt(vx^2 + vy^2 + vz^2)
    if v > p.v_max
        return false
    end
    
    return true
end

## Chance constraint implementation

"""
Constraint function: g(s) ≤ 0 for feasibility.

Constraints:
  1-3: Stay out of dangerous zones (one constraint per zone)
  4-5: Altitude bounds (z ≥ z_min and z ≤ z_max)
  6-7: Horizontal bounds (|x| ≤ xy_max, |y| ≤ xy_max)
  8: Velocity bound (v ≤ v_max)

For dangerous zones: g = radius - distance_to_center ≤ 0
  (i.e., distance_to_center ≥ radius means safe)
"""
function constraint_func(p::HelicopterMDP, s::AbstractVector)
    x, y, z, vx, vy, vz = s
    
    constraints = Float64[]
    
    # Dangerous zone constraints (stay OUT of zones)
    # g = radius - distance ≤ 0  =>  distance ≥ radius (safe)
    for zone in p.danger_zones
        cx, cy, cz, r = zone
        dist = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
        push!(constraints, r - dist)  # Safe when negative (dist > radius)
    end
    
    # Altitude constraints
    push!(constraints, p.z_min - z)      # z ≥ z_min
    push!(constraints, z - p.z_max)      # z ≤ z_max
    
    # Horizontal boundary constraints
    push!(constraints, x - p.xy_max)     # x ≤ xy_max
    push!(constraints, -x - p.xy_max)    # x ≥ -xy_max
    push!(constraints, y - p.xy_max)     # y ≤ xy_max
    push!(constraints, -y - p.xy_max)    # y ≥ -xy_max
    
    # Velocity constraint
    v = sqrt(vx^2 + vy^2 + vz^2)
    push!(constraints, v - p.v_max)      # v ≤ v_max
    
    return constraints
end

"""
Jacobian of constraint function: ∂g/∂s.
Each row is the gradient of one constraint w.r.t. state [x, y, z, vx, vy, vz].
"""
function constraint_jacobian(p::HelicopterMDP, s::AbstractVector)
    x, y, z, vx, vy, vz = s
    
    jac = zeros(num_constraints(p), 6)
    idx = 1
    
    # Dangerous zone constraints
    for zone in p.danger_zones
        cx, cy, cz, r = zone
        dist = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
        dist = max(dist, 1e-6)  # Avoid division by zero
        
        # ∂g/∂x = -∂dist/∂x = -(x - cx)/dist
        jac[idx, 1] = -(x - cx) / dist  # ∂g/∂x
        jac[idx, 2] = -(y - cy) / dist  # ∂g/∂y
        jac[idx, 3] = -(z - cz) / dist  # ∂g/∂z
        # Velocity components don't affect distance
        
        idx += 1
    end
    
    # Altitude constraints
    jac[idx, 3] = -1.0  # ∂(z_min - z)/∂z
    idx += 1
    jac[idx, 3] = 1.0   # ∂(z - z_max)/∂z
    idx += 1
    
    # Horizontal boundary constraints
    jac[idx, 1] = 1.0   # ∂(x - xy_max)/∂x
    idx += 1
    jac[idx, 1] = -1.0  # ∂(-x - xy_max)/∂x
    idx += 1
    jac[idx, 2] = 1.0   # ∂(y - xy_max)/∂y
    idx += 1
    jac[idx, 2] = -1.0  # ∂(-y - xy_max)/∂y
    idx += 1
    
    # Velocity constraint
    v = sqrt(vx^2 + vy^2 + vz^2)
    v = max(v, 1e-6)  # Avoid division by zero
    jac[idx, 4] = vx / v  # ∂v/∂vx
    jac[idx, 5] = vy / v  # ∂v/∂vy
    jac[idx, 6] = vz / v  # ∂v/∂vz
    
    return jac
end

num_constraints(p::HelicopterMDP) = length(p.danger_zones) + 7

"""
Visualize the helicopter state in 3D.
Returns the helicopter position for plotting.
"""
function visualize(p::HelicopterMDP, s::AbstractVector)
    x, y, z, vx, vy, vz = s
    return [x, y, z]
end

"""
Check if the helicopter has reached a waypoint.
"""
function reached_waypoint(p::HelicopterMDP, s::AbstractVector, waypoint::AbstractVector, tolerance=1.0)
    x, y, z = s[1:3]
    wx, wy, wz = waypoint
    dist = sqrt((x - wx)^2 + (y - wy)^2 + (z - wz)^2)
    return dist <= tolerance
end

"""
Check if the helicopter has reached the goal (landing site).
"""
function reached_goal(p::HelicopterMDP, s::AbstractVector, tolerance=2.0)
    goal_pos = p.s_goal[1:3]
    return reached_waypoint(p, s, goal_pos, tolerance)
end
