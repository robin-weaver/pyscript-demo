import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
m = 0.04593  # mass of the golf ball
g = 9.81  # acceleration due to gravity
Cd = 0.3  # drag coefficient
Cl = 0.2  # lift coefficient
rho = 1.205  # air density
R = 0.0214  # radius of the golf ball
A = np.pi * R**2  # cross-sectional area of the golf ball
kD = 0.5 * Cd * rho * A  # drag constant
kL = 0.5 * Cl * rho * A  # lift constant


def model(
        v0=44,  # initial vel, m/s
        alpha=55,  # launch angle, deg
        omega=(0, 0, 0)  # vector of angular vel, rad/s (x, y, z)
):
    alpha = np.radians(alpha)  # initial angle
    omega = np.array(omega)  # initial angular velocity
    x0, y0, z0 = 0, 0, 0  # initial position
    vx0, vy0, vz0 = v0 * np.cos(alpha), 0, v0 * np.sin(alpha)  # initial velocity components
    state0 = [x0, y0, z0, vx0, vy0, vz0]  # initial state vector

    # Differential equations
    def deriv(state, t):
        x, y, z, vx, vy, vz = state
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        FD = -kD * v * np.array([vx, vy, vz])
        FM = kL * np.cross(omega, [vx, vy, vz])
        G = np.array([0, 0, -m*g])
        dvdt = (FD + FM + G) / m
        dxdt = [vx, vy, vz]
        return dxdt + list(dvdt)


    # Time array
    t = np.linspace(0, 15, num=1000)

    # Solve the differential equations
    sol = odeint(deriv, state0, t)
    sol1 = []

    for i, vec in enumerate(sol):
        if vec[2] >= 0:
            sol1.append(vec)
    return np.array(sol1)


results = model()
# Plot the trajectory in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(results[:, 0], results[:, 1], results[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('auto')

display(plt, target="plot")

form = Element("#app-form")

# @pydom.when(form, 'change')
# def sim():
#     pass