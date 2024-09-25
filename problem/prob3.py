import numpy as np
import matplotlib.pyplot as plt

# Define a range of small angle values in radians (from 0 to 0.5 rad)
delta_vals = np.linspace(0, 0.5, 100)

# Calculate sin(δ) and cos(δ)
sin_delta = np.sin(delta_vals)
cos_delta = np.cos(delta_vals)

# Approximate small angle values
sin_approx = delta_vals
cos_approx = np.ones_like(delta_vals)

# Plot sin(δ) vs δ (Approximation)
plt.figure(figsize=(10, 6))
plt.plot(delta_vals, sin_delta, label="sin(δ)", color='b')
plt.plot(delta_vals, sin_approx, '--', label="δ (Approximation)", color='r')
plt.xlabel("δ (radians)")
plt.ylabel("sin(δ)")
plt.title("Validation of sin(δ) ≈ δ for Small Angles")
plt.legend()
plt.grid(True)
plt.show()

# Plot cos(δ) vs 1 (Approximation)
plt.figure(figsize=(10, 6))
plt.plot(delta_vals, cos_delta, label="cos(δ)", color='g')
plt.plot(delta_vals, cos_approx, '--', label="1 (Approximation)", color='r')
plt.xlabel("δ (radians)")
plt.ylabel("cos(δ)")
plt.title("Validation of cos(δ) ≈ 1 for Small Angles")
plt.legend()
plt.grid(True)
plt.show()

# Calculate the error for each approximation
sin_error = np.abs(sin_delta - sin_approx)
cos_error = np.abs(cos_delta - cos_approx)

# Plot the error in sin(δ) approximation
plt.figure(figsize=(10, 6))
plt.plot(delta_vals, sin_error, label="sin(δ) Error", color='b')
plt.xlabel("δ (radians)")
plt.ylabel("Error")
plt.title("Error in sin(δ) ≈ δ Approximation for Small Angles")
plt.legend()
plt.grid(True)
plt.show()

# Plot the error in cos(δ) approximation
plt.figure(figsize=(10, 6))
plt.plot(delta_vals, cos_error, label="cos(δ) Error", color='g')
plt.xlabel("δ (radians)")
plt.ylabel("Error")
plt.title("Error in cos(δ) ≈ 1 Approximation for Small Angles")
plt.legend()
plt.grid(True)
plt.show()
