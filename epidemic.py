import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions
N = 1000
S0, I0, R0 = 1000, 10, 0 
y0 = S0, I0, R0

# Parameters
beta = 0.5
gamma = 0.1

# Solve ODE system
t = np.linspace(0, 100, 100)
y = odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = y.T

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('COVID-19 SIR Model')
plt.legend()
plt.show()