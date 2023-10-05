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
def_el = 20
def_az = -55


def model(v0=50.0, alpha=12.5, omega=(0, -34, 0)):
    alpha = np.radians(alpha)  # initial angle
    omega = np.array(omega)  # initial angular velocity
    x0, y0, z0 = 0, 0, 0  # initial position
    vx0, vy0, vz0 = v0 * np.cos(alpha), 0, v0 * np.sin(alpha)  # initial velocity components
    state0 = [x0, y0, z0, vx0, vy0, vz0]  # initial state vector

    def deriv(state, t):
        x, y, z, vx, vy, vz = state
        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        FD = -kD * v * np.array([vx, vy, vz])
        FM = kL * np.cross(omega, [vx, vy, vz])
        G = np.array([0, 0, -m * g])
        dvdt = (FD + FM + G) / m
        dxdt = [vx, vy, vz]
        return dxdt + list(dvdt)

    t = np.linspace(0, 15, num=1000)
    sol = odeint(deriv, state0, t)
    return np.array([vec for vec in sol if vec[2] >= 0])


def plot_trajectory(res, **kwargs):
    ax.plot(res[:, 0], res[:, 1], res[:, 2], **kwargs)


def simulate():
    try:
        vel = float(Element('speed').value)
        la = float(Element('angle').value)
        av = (int(Element('x-spin').value), int(Element('y-spin').value), -int(Element('z-spin').value))
        mav = (-int(Element('x-spin').value), int(Element('y-spin').value), int(Element('z-spin').value))
        res = model(vel, la, av)
        res1 = model(vel, la, mav)
        plot_trajectory(res1, label='', color='white', linestyle='-', alpha=0.001)
        plot_trajectory(res)
        refresh_plot()
    except Exception as e:
        print(str(e))


def refresh_plot():
    js.document.getElementById('plot').innerHTML = ''
    ax.view_init(elev=def_el, azim=def_az)
    display(plt, target="plot")


def clear():
    global ax
    del ax
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    refresh_plot()


def rotate_view(elevation_change=0, azimuth_change=0):
    global def_el, def_az
    def_el += elevation_change
    def_az += azimuth_change
    refresh_plot()


def rleft():
    rotate_view(azimuth_change=-5)


def rright():
    rotate_view(azimuth_change=5)


def rup():
    rotate_view(elevation_change=5)


def rdown():
    rotate_view(elevation_change=-5)


results = model()
results1 = model(omega=(0, -20, -30))
results2 = model(v0=38, alpha=55, omega=(0, -5, 0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_trajectory(results, label='Normal(ish) shot', color='green')
plot_trajectory(results1, label='Slice', color='orange')
plot_trajectory(results2, label='Wedge shot', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_box_aspect([1,1,1])
ax.view_init(elev=def_el, azim=def_az)
refresh_plot()