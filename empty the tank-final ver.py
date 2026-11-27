# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:25:31 2025

@author: VK
"""


import numpy as np
import matplotlib.pyplot as plt

def tank_drain(h0, At, Ao, Cd=0.62, g=9.81, dt=0.1):
    """
    Simulate draining of a tank using Torricelli's law (numerical Euler method).
    """
    k = (Cd * Ao / At) * np.sqrt(2 * g)
    
    t_vals = [0.0]
    h_vals = [h0]
    t = 0.0
    h = h0

    while h > 0:
        dhdt = -k * np.sqrt(h)
        h += dhdt * dt
        h = max(h, 0)
        t += dt
        t_vals.append(t)
        h_vals.append(h)

    return np.array(t_vals), np.array(h_vals), k


# Example simulation for your case
if __name__ == "__main__":
    # Tank parameters
    D_tank = 1.0       # tank diameter [m]
    h0 = 1.0           # initial liquid height [m]
    d_out = 0.025       # outlet diameter [m]

    # Areas
    At = np.pi * (D_tank/2)**2     # tank cross-sectional area [m^2]
    Ao = np.pi * (d_out/2)**2      # orifice area [m^2]

    Cd = 0.7                      # discharge coefficient
    g = 9.81                       # gravity [m/s^2]

    # Run numerical simulation
    t, h, k = tank_drain(h0, At, Ao, Cd, g, dt=0.05)

    # Analytical solution for h(t)
    t_empty = 2*np.sqrt(h0)/k
    t_analytical = np.linspace(0, t_empty, 200)
    h_analytical = (np.sqrt(h0) - 0.5*k*t_analytical)**2

    # Plot results
    plt.figure(figsize=(8,5))
    plt.plot(t, h, "b-", lw=2, label="Numerical (Euler)")
    plt.plot(t_analytical, h_analytical, "r--", lw=2, label="Analytical")
    plt.xlabel("Time [s]")
    plt.ylabel("Water height h(t) [m]")
    plt.title("Draining of a Tank (D=1m, h0=1m, outlet=25mm)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print emptying time
    print(f"Tank empties in approx {t_empty:.2f} seconds (analytical)")
    print(f"Tank empties in approx {t[-1]:.2f} seconds (numerical)")
