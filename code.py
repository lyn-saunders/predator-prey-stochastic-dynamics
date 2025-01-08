# import required packages 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Part 2 - Deterministic Simulations

# For the parameter combination gamma_p = 1, mu_p = 0.01, gamma_h = 1, mu_h = 100, K = 200 and the initial condition p(0) = 100, h(0) = 45, solve the system numerically on the time interval [0,30]. Plot the solution time series for h and p. Do the simulations agree with our calculations? 

# Define parameters of equations (1) and (2) and their assigned values

gamma_p = 1
mu_p = 0.01
gamma_h = 1
mu_h = 100
K = 200

# Define the system of ODEs

def predator_prey_system(y, t):

    # p represents the prey
    # h represents the predator
    # y will store both p and h values

    p, h = y 
    dp_dt = gamma_p*p*(1 - p/K) - mu_p*h*p 
    dh_dt = gamma_h*p*h - mu_h*h
    return [dp_dt, dh_dt]

# Initial conditions (prey and predator densities at time t = 0)

p0 = 100
h0 = 45
y0 = [p0, h0]

# Initialise time vector for the solution (from 0 to 30, with 300 points)

t = np.linspace(0, 30, 300)

# Solve the system of ODEs (using odeint from scipy.integrate)

solution = odeint(predator_prey_system, y0, t)

# Transposing the solution matrix to separate corresponding p and h values

p, h = solution.T 

# Plot the solution (as two separate plots)

plt.figure(figsize=(12, 5))

# Plot prey (p) population

plt.subplot(1, 2, 1)
plt.plot(t, p, label='Prey (p(t))', color='#FF69B4')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Prey Population Over Time')
plt.legend()

# Plot predator (h) population

plt.subplot(1, 2, 2)
plt.plot(t, h, label='Predator (h(t))', color='purple')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator Population Over Time')
plt.legend()

plt.tight_layout()
plt.show()


# Part 3 - Stochastic Simulations

# Simulate the SDEs (3) and (4) numerically, using a timestep of dt = 0.0001 and alpha_p = alpha_h = 5. Use reflecting boundary conditions at p = 0 and at h = 0 to ensure non-negative population sizes. How do the simulated dynamics compare to our previously plotted solutions of ODEs?

# Define parameters and their assigned values

# alpha_p and alpha_h are the coefficients of the noise terms in the stochastic differential equations for p and h

alpha_p = 5
alpha_h = 5
dt = 0.0001  
total_steps = int(30 / dt) 
t_vec = np.linspace(0, 30, total_steps) 

# Initialise arrays to store the prey and predator populations over time steps

p_vec = np.zeros(total_steps)
h_vec = np.zeros(total_steps)
p_vec[0] = p0 
h_vec[0] = h0 

# Simulate the SDEs

for i in range(1, total_steps):
    p = p_vec[i - 1]
    h = h_vec[i - 1]

    # Deterministic terms of the SDEs

    dp_dt = gamma_p*p*(1 - p/K) - mu_p*h*p
    dh_dt = gamma_h*p*h - mu_h*h

    # Noise terms of the SDEs

    noise_p = alpha_p*np.random.normal(0, np.sqrt(dt))
    noise_h = alpha_h*np.random.normal(0, np.sqrt(dt))

    # Update populations through time

    p_new = p + dp_dt*dt + noise_p
    h_new = h + dh_dt*dt + noise_h

    # Apply reflecting boundary conditions for biologically relevent (non-negative) populations

    p_vec[i] = max(p_new, 0)
    h_vec[i] = max(h_new, 0)

# Plot the simulation results

plt.figure(figsize=(12, 5))

# Prey (p) population

plt.subplot(1, 2, 1)
plt.plot(t_vec, p_vec, label='Prey (p(t)) with Noise', color='#FF69B4')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Prey Population Over Time (SDE)')
plt.legend()

# Predator (h) population

plt.subplot(1, 2, 2)
plt.plot(t_vec, h_vec, label='Predator (h(t)) with Noise', color='purple')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator Population Over Time (SDE)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot both prey and predator populations on the same plot

plt.figure(figsize=(10, 6))

plt.plot(t_vec, p_vec, label='Prey (p(t)) with Noise', color='#FF69B4')
plt.plot(t_vec, h_vec, label='Predator (h(t)) with Noise', color='purple')
plt.xlabel('Time')
plt.ylabel('Population Density')
plt.title('Population Densities of Prey and Predator Over Time')
plt.legend()

plt.show()

# Part 6 - The Power Spectrum

# For the parameter values previously given, plot S_p(omega) on the domain omega in [6, 8]. Does the power spectrum have a peak? What is the frequency omega_0 at which this peak occurs and what does it represent? Consider the value T_p = 2*pi/omega_0. What does T_p represent?

# Define the function (along with the new parameters) to calculate the power spectrum of the (linearised) prey dynamics

def power_spectrum_prey(omega, gamma_p, mu_p, gamma_h, mu_h, alpha_p, alpha_h, p0, h0, K):
    
    # Define the complex number i

    i = 1j

    # Define our simplified parameters

    x_p = gamma_p*(1 - (2*p0/K)) - mu_p*h0
    x_h = gamma_h*h0
    y_p = -mu_p*p0
    y_h = gamma_h*p0 - mu_h

    # Simplify our noise coefficients (squared)

    prey_noise_coef = alpha_p**2
    predator_noise_coef = alpha_h**2

    # Initialise our matrix A, and find the determinant

    A_matrix = np.array([[i*omega - x_p, -y_p], [-x_h, i*omega - y_h]])
    det_A = np.linalg.det(A_matrix)

    # Define the numerator and denominator terms needed for our final power spectrum calculation

    num = (prey_noise_coef*(omega**2 + y_h**2) + predator_noise_coef*y_p**2)
    den = (np.abs(det_A))**2
    S_p = num / den

    return S_p

# Store the power and omega values during the simulation

omega_values = np.linspace(6, 8, 500)
S_p_values = [power_spectrum_prey(omega, gamma_p, mu_p, gamma_h, mu_h, alpha_p, alpha_h, p0, h0, K) for omega in omega_values]

# Plotting the prey power spectrum S_p(omega) against omega values

plt.figure(figsize=(10, 6))
plt.plot(omega_values, S_p_values, label='Power spectrum of prey (p)', color='#FF69B4')
plt.xlabel('Frequency (ω)')
plt.ylabel('Power Spectrum S_p(ω)')
plt.title('Power Spectrum of Prey Population')
plt.legend()
plt.grid(True)

plt.show()

# Find peak of power spectrum at omega_0

peak_index = np.argmax(S_p_values)
omega_0 = omega_values[peak_index]
peak_value = S_p_values[peak_index]

print(f"Peak found at ω_0 = {omega_0:.4f}, S_p(ω_0) = {peak_value:.4f}")

