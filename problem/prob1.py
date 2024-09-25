import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
H = 4
Pm = 1
f = 60
omega_s = 2 * np.pi * f  # Synchronous speed in rad/s

#a. Swing equation as a system of first-order ODEs
def swing_eq(t, y):
    delta, omega = y
    Pe = 2 * np.sin(delta)  # Electrical power
    d_delta = omega_s * omega
    d_omega = (Pm - Pe) / (2 * H)
    return [d_delta, d_omega]

# Initial conditions: [delta(0), omega(0)]
y0 = [0, 0]
# Time array
t_span = [0, 4]
t_eval = np.linspace(0, 4, 1000)

# Solve using solve_ivp
sol = solve_ivp(swing_eq, t_span, y0, t_eval=t_eval, method='RK45')

# Plot
plt.plot(sol.t, sol.y[0], label="Delta (δ)")
plt.plot(sol.t, sol.y[1], label="Omega (ω)")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation Solution using solve_ivp (ode45)")
plt.legend()
plt.grid(True)
plt.show()



#b. Forward Euler method
def forward_euler(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    for i in range(1, len(t)):
        Pe = 2 * np.sin(delta[i-1])
        delta[i] = delta[i-1] + dt * omega_s * omega[i-1]
        omega[i] = omega[i-1] + dt * (Pm - Pe) / (2 * H)
    
    return t, delta, omega

# Parameters
dt = 0.005
t_end = 4

t, delta_fwd, omega_fwd = forward_euler(0, 0, dt, t_end)

# Plot
plt.plot(t, delta_fwd, label="Delta (δ)")
plt.plot(t, omega_fwd, label="Omega (ω)")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Forward Euler Method")
plt.legend()
plt.grid(True)
plt.show()


#c. Backward Euler method would involve solving implicit equations.
# Here is a simple approximation-based implementation for small time steps.
def backward_euler(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    for i in range(1, len(t)):
        Pe = 2 * np.sin(delta[i-1])
        omega_new = omega[i-1] + dt * (Pm - Pe) / (2 * H)
        delta_new = delta[i-1] + dt * omega_s * omega_new
        delta[i], omega[i] = delta_new, omega_new
    
    return t, delta, omega

t, delta_bwd, omega_bwd = backward_euler(0, 0, dt, t_end)

# Plot
plt.plot(t, delta_bwd, label="Delta (δ)")
plt.plot(t, omega_bwd, label="Omega (ω)")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Backward Euler Method")
plt.legend()
plt.grid(True)
plt.show()


#d. Trapezoidal Rule method for solving ODEs
def trapezoidal(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    for i in range(1, len(t)):
        Pe_prev = 2 * np.sin(delta[i-1])
        omega_mid = omega[i-1] + dt / 2 * (Pm - Pe_prev) / (2 * H)
        delta_mid = delta[i-1] + dt / 2 * omega_s * omega[i-1]
        Pe_mid = 2 * np.sin(delta_mid)
        omega[i] = omega[i-1] + dt * (Pm - Pe_mid) / (2 * H)
        delta[i] = delta[i-1] + dt * omega_s * omega_mid
    
    return t, delta, omega

t, delta_trap, omega_trap = trapezoidal(0, 0, dt, t_end)

# Plot
plt.plot(t, delta_trap, label="Delta (δ)")
plt.plot(t, omega_trap, label="Omega (ω)")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Trapezoidal Rule Method")
plt.legend()
plt.grid(True)
plt.show()

#e. Euler's Full-Step Modification Method 
def euler_full_step(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0

    for i in range(1, len(t)):
        Pe = 2 * np.sin(delta[i-1])
        # Full-step prediction for delta and omega
        delta[i] = delta[i-1] + dt * omega_s * omega[i-1]
        omega[i] = omega[i-1] + dt * (Pm - Pe) / (2 * H)

    return t, delta, omega

# Parameters for Euler's Full-Step Method
dt = 0.005
t_end = 4

# Apply Euler's Full-Step Method
t, delta_full_step, omega_full_step = euler_full_step(0, 0, dt, t_end)

# Plot for Euler’s Full-Step Method (without damping)
plt.figure(figsize=(10, 6))
plt.plot(t, delta_full_step, label="Delta (δ) - Full-Step")
plt.plot(t, omega_full_step, label="Omega (ω) - Full-Step")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation using Euler’s Full-Step Method (without damping)")
plt.legend()
plt.grid(True)
plt.show()



#f. Euler’s Half-Step Modification Method
def euler_half_step(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    for i in range(1, len(t)):
        Pe = 2 * np.sin(delta[i-1])
        # Half-step prediction
        delta_half = delta[i-1] + 0.5 * dt * omega_s * omega[i-1]
        omega_half = omega[i-1] + 0.5 * dt * (Pm - Pe) / (2 * H)
        
        # Full-step correction
        Pe_half = 2 * np.sin(delta_half)
        delta[i] = delta[i-1] + dt * omega_s * omega_half
        omega[i] = omega[i-1] + dt * (Pm - Pe_half) / (2 * H)
    
    return t, delta, omega

# Parameters for Euler's Half-Step Method
dt = 0.005
t_end = 4

# Apply Euler's Half-Step Method
t, delta_half_step, omega_half_step = euler_half_step(0, 0, dt, t_end)

# Plot for Euler’s Half-Step Method
plt.figure(figsize=(10, 6))
plt.plot(t, delta_half_step, label="Delta (δ) - Half-Step")
plt.plot(t, omega_half_step, label="Omega (ω) - Half-Step")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation using Euler’s Half-Step Method")
plt.legend()
plt.grid(True)
plt.show()


#g. Adam-Bashforth Second-Order Method
def adam_bashforth_second_order(delta_0, omega_0, dt, t_end):
    t = np.arange(0, t_end, dt)
    delta = np.zeros(len(t))
    omega = np.zeros(len(t))
    delta[0], omega[0] = delta_0, omega_0
    
    # Use Forward Euler for the first step to start Adam-Bashforth
    Pe = 2 * np.sin(delta[0])
    delta[1] = delta[0] + dt * omega_s * omega[0]
    omega[1] = omega[0] + dt * (Pm - Pe) / (2 * H)
    
    for i in range(2, len(t)):
        Pe_prev = 2 * np.sin(delta[i-2])
        Pe_curr = 2 * np.sin(delta[i-1])
        delta[i] = delta[i-1] + dt * omega_s * (3/2 * omega[i-1] - 1/2 * omega[i-2])
        omega[i] = omega[i-1] + dt * ((3/2 * (Pm - Pe_curr) - 1/2 * (Pm - Pe_prev)) / (2 * H))
    
    return t, delta, omega

# Apply Adam-Bashforth Second-Order Method
t, delta_ab2, omega_ab2 = adam_bashforth_second_order(0, 0, dt, t_end)

# Plot for Adam-Bashforth Second-Order Method
plt.figure(figsize=(10, 6))
plt.plot(t, delta_ab2, label="Delta (δ) - Adam-Bashforth 2nd Order")
plt.plot(t, omega_ab2, label="Omega (ω) - Adam-Bashforth 2nd Order")
plt.xlabel("Time [s]")
plt.ylabel("Delta / Omega")
plt.title("Swing Equation using Adam-Bashforth Second-Order Method")
plt.legend()
plt.grid(True)
plt.show()

