import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pyscript import Element, display
import js

# Constants
m = 0.04593  # mass of the golf ball
g = 9.81  # acceleration due to gravity
Cd = 0.3  # drag coefficient
Cl = 0.2  # lift coefficient
rho = 1.205  # air density
R = 0.0214  # radius of the golf ball
A = np.pi * R ** 2  # cross-sectional area of the golf ball
kD = 0.5 * Cd * rho * A  # drag constant
kL = 0.5 * Cl * rho * A  # lift constant


def model(
        v0=50.0,  # initial vel, m/s
        alpha=12.5,  # launch angle, deg
        omega=(0, -34, 0)  # vector of angular vel, rad/s (x, y, z)
):
    alpha = np.radians(alpha)  # initial angle
    omega = np.array(omega)  # initial angular velocity
    x0, y0, z0 = 0, 0, 0  # initial position
    vx0, vy0, vz0 = v0 * np.cos(alpha), 0, v0 * np.sin(alpha)  # initial velocity components
    state0 = [x0, y0, z0, vx0, vy0, vz0]  # initial state vector

    # Differential equations
    def deriv(state, t):
        x, y, z, vx, vy, vz = state
        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        FD = -kD * v * np.array([vx, vy, vz])
        FM = kL * np.cross(omega, [vx, vy, vz])
        G = np.array([0, 0, -m * g])
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
results1 = model(omega=(0, -20, -30))
results2 = model(v0=38, alpha=55, omega=(0, -5, 0))


# Plot the trajectory in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(results[:, 0], results[:, 1], results[:, 2], label='Normal(ish) shot', color='green')
ax.plot(results1[:, 0], results1[:, 1], results1[:, 2], label='Slice', color='orange')
ax.plot(results2[:, 0], results2[:, 1], results2[:, 2], label='Wedge shot', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_box_aspect([1,1,1])
ax.view_init(elev=20, azim=-55)
display(plt, target="plot")


def simulate():
    try:
        vel = float(Element('speed').value)
        la = float(Element('angle').value)
        av = (float(Element('x-spin').value), int(Element('y-spin').value), -int(Element('z-spin').value))
        mav = (-float(Element('x-spin').value), int(Element('y-spin').value), int(Element('z-spin').value))
        res = model(vel, la, av)
        res1 = model(vel, la, mav)
        ax.plot(res1[:, 0], res1[:, 1], res1[:, 2], linestyle='-', color='white', alpha=0.001)
        ax.plot(res[:, 0], res[:, 1], res[:, 2])

        js.document.getElementById('plot').innerHTML = ''
        display(plt, target="plot")
    except Exception as e:
        print(str(e))


def_az = -55
def_el = 20


def clear():
    global ax
    del ax
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    js.document.getElementById('plot').innerHTML = ''
    ax.view_init(elev=def_el, azim=def_az)
    display(plt, target="plot")





def rleft():
    global def_az
    def_az = def_az - 5
    js.document.getElementById('plot').innerHTML = ''
    ax.view_init(elev=def_el, azim=def_az)
    display(plt, target="plot")
    pass


def rright():
    global def_az
    def_az = def_az + 5
    js.document.getElementById('plot').innerHTML = ''
    ax.view_init(elev=def_el, azim=def_az)
    display(plt, target="plot")


def rup():
    global def_el
    def_el = def_el + 5
    js.document.getElementById('plot').innerHTML = ''
    ax.view_init(elev=def_el, azim=def_az)
    display(plt, target="plot")


def rdown():
    global def_el
    def_el = def_el - 5
    js.document.getElementById('plot').innerHTML = ''
    ax.view_init(elev=def_el, azim=def_az)
    display(plt, target="plot")