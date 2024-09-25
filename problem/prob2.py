import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
H = 4
Pm = 1
f = 60
D = 1  # Damping coefficient
omega_s = 2 * np.pi * f  # Synchronous speed in rad/s

# a. Swing equation with damping (solve_ivp equivalent to ode45)
def swing_eq_with_damping(t, y):
    delta, omega = y
    Pe = 2 * np.sin(delta)  # Electrical power
    d_delta = omega_s * omega
    d_omega = (Pm - Pe - D * omega) / (2 * H)
    return [d_delta, d_omega]

# Initial conditions: [delta(0), omega(0)]
y0 = [0, 0]
# Time array
t_span = [0, 4]
t_eval = np.linspace(0, 4, 1000)

# Solve using solve_ivp (ode45 equivalent)
sol_with_damping = solve_ivp(swing_eq_with_damping, t_span, y0, t_eval=t_eval, method='RK45')

# Plot the results for solve_ivp (ode45 equivalent)
plt.figure(figsize=(10, 6))
plt.plot(sol_with_damping.t, sol_with_damping.y[0], label="Delta (δ) - ode45")
plt.plot(sol_with_damping.t, sol_with_damping.y[1], label="Omega (ω) - ode45")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation with Damping using solve_ivp (ode45)")
plt.legend()
plt.grid(True)
plt.show()


# b. Forward Euler's Method with Damping
def forward_euler_damping(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    for i in range(1, len(t)):
        Pe = 2 * np.sin(delta[i-1])
        delta[i] = delta[i-1] + dt * omega_s * omega[i-1]
        omega[i] = omega[i-1] + dt * (Pm - Pe - D * omega[i-1]) / (2 * H)
    
    return t, delta, omega

# Parameters
dt = 0.005
t_end = 4

# Forward Euler with damping
t, delta_fwd, omega_fwd = forward_euler_damping(0, 0, dt, t_end)

# Plot the results for Forward Euler Method
plt.figure(figsize=(10, 6))
plt.plot(t, delta_fwd, label="Delta (δ) - Forward Euler")
plt.plot(t, omega_fwd, label="Omega (ω) - Forward Euler")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation with Damping using Forward Euler Method")
plt.legend()
plt.grid(True)
plt.show()


# c. Backward Euler's Method with Damping
def backward_euler_damping(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    for i in range(1, len(t)):
        Pe = 2 * np.sin(delta[i-1])
        omega_new = omega[i-1] + dt * (Pm - Pe - D * omega[i-1]) / (2 * H)
        delta_new = delta[i-1] + dt * omega_s * omega_new
        delta[i], omega[i] = delta_new, omega_new
    
    return t, delta, omega

# Backward Euler with damping
t, delta_bwd, omega_bwd = backward_euler_damping(0, 0, dt, t_end)

# Plot the results for Backward Euler Method
plt.figure(figsize=(10, 6))
plt.plot(t, delta_bwd, label="Delta (δ) - Backward Euler")
plt.plot(t, omega_bwd, label="Omega (ω) - Backward Euler")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation with Damping using Backward Euler Method")
plt.legend()
plt.grid(True)
plt.show()


# d. Trapezoidal Rule Method with Damping
def trapezoidal_damping(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    for i in range(1, len(t)):
        Pe_prev = 2 * np.sin(delta[i-1])
        omega_mid = omega[i-1] + dt / 2 * (Pm - Pe_prev - D * omega[i-1]) / (2 * H)
        delta_mid = delta[i-1] + dt / 2 * omega_s * omega[i-1]
        Pe_mid = 2 * np.sin(delta_mid)
        omega[i] = omega[i-1] + dt * (Pm - Pe_mid - D * omega_mid) / (2 * H)
        delta[i] = delta[i-1] + dt * omega_s * omega_mid
    
    return t, delta, omega

# Trapezoidal Rule with damping
t, delta_trap, omega_trap = trapezoidal_damping(0, 0, dt, t_end)

# Plot the results for Trapezoidal Rule Method
plt.figure(figsize=(10, 6))
plt.plot(t, delta_trap, label="Delta (δ) - Trapezoidal Rule")
plt.plot(t, omega_trap, label="Omega (ω) - Trapezoidal Rule")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation with Damping using Trapezoidal Rule Method")
plt.legend()
plt.grid(True)
plt.show()


#e. Euler's Full-Step Modification Method (with damping)
def euler_full_step_damping(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0

    for i in range(1, len(t)):
        Pe = 2 * np.sin(delta[i-1])
        # Full-step prediction for delta and omega with damping
        delta[i] = delta[i-1] + dt * omega_s * omega[i-1]
        omega[i] = omega[i-1] + dt * (Pm - Pe - D * omega[i-1]) / (2 * H)

    return t, delta, omega

# Apply Euler's Full-Step Method with Damping
t, delta_full_step_damping, omega_full_step_damping = euler_full_step_damping(0, 0, dt, t_end)

# Plot for Euler’s Full-Step Method (with damping)
plt.figure(figsize=(10, 6))
plt.plot(t, delta_full_step_damping, label="Delta (δ) - Full-Step with Damping")
plt.plot(t, omega_full_step_damping, label="Omega (ω) - Full-Step with Damping")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation with Damping using Euler’s Full-Step Method")
plt.legend()
plt.grid(True)
plt.show()
 


#f. Euler’s Half-Step Method with Damping
def euler_half_step_damping(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    for i in range(1, len(t)):
        Pe = 2 * np.sin(delta[i-1])
        # Half-step prediction
        delta_half = delta[i-1] + 0.5 * dt * omega_s * omega[i-1]
        omega_half = omega[i-1] + 0.5 * dt * (Pm - Pe - D * omega[i-1]) / (2 * H)
        
        # Full-step correction
        Pe_half = 2 * np.sin(delta_half)
        delta[i] = delta[i-1] + dt * omega_s * omega_half
        omega[i] = omega[i-1] + dt * (Pm - Pe_half - D * omega_half) / (2 * H)
    
    return t, delta, omega

# Apply Euler's Half-Step Method with Damping
t, delta_half_step_damping, omega_half_step_damping = euler_half_step_damping(0, 0, dt, t_end)

# Plot for Euler’s Half-Step Method with Damping
plt.figure(figsize=(10, 6))
plt.plot(t, delta_half_step_damping, label="Delta (δ) - Half-Step with Damping")
plt.plot(t, omega_half_step_damping, label="Omega (ω) - Half-Step with Damping")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation with Damping using Euler’s Half-Step Method")
plt.legend()
plt.grid(True)
plt.show()


#g. Adam-Bashforth Second-Order Method with Damping
def adam_bashforth_second_order_damping(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    # Use Forward Euler for the first step to start Adam-Bashforth
    Pe = 2 * np.sin(delta[0])
    delta[1] = delta[0] + dt * omega_s * omega[0]
    omega[1] = omega[0] + dt * (Pm - Pe - D * omega[0]) / (2 * H)
    
    for i in range(2, len(t)):
        Pe_prev = 2 * np.sin(delta[i-2])
        Pe_curr = 2 * np.sin(delta[i-1])
        delta[i] = delta[i-1] + dt * omega_s * (3/2 * omega[i-1] - 1/2 * omega[i-2])
        omega[i] = omega[i-1] + dt * ((3/2 * (Pm - Pe_curr - D * omega[i-1]) 
                                      - 1/2 * (Pm - Pe_prev - D * omega[i-2])) / (2 * H))
    
    return t, delta, omega

# Apply Adam-Bashforth Second-Order Method with Damping
t, delta_ab2_damping, omega_ab2_damping = adam_bashforth_second_order_damping(0, 0, dt, t_end)

# Plot for Adam-Bashforth Second-Order Method with Damping
plt.figure(figsize=(10, 6))
plt.plot(t, delta_ab2_damping, label="Delta (δ) - Adam-Bashforth 2nd Order with Damping")
plt.plot(t, omega_ab2_damping, label="Omega (ω) - Adam-Bashforth 2nd Order with Damping")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation with Damping using Adam-Bashforth Second-Order Method")
plt.legend()
plt.grid(True)
plt.show()
