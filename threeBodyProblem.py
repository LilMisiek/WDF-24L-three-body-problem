import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype='float64')
        self.velocity = np.array(velocity, dtype='float64')

class ThreeBodySystem:
    def __init__(self, body1, body2, body3, G=1):
        self.bodies = [body1, body2, body3]
        self.G = G

    def compute_accelerations(self):
        accelerations = []
        for i, body in enumerate(self.bodies):
            acc = np.zeros(2)
            for j, other_body in enumerate(self.bodies):
                if i != j:
                    r = other_body.position - body.position
                    distance = np.linalg.norm(r)
                    if distance == 0:
                        continue  # Unikamy dzielenia przez zero
                    acc += self.G * other_body.mass * r / distance**3
            accelerations.append(acc)
        return accelerations

    def step(self, dt):
        positions = np.array([body.position for body in self.bodies])
        velocities = np.array([body.velocity for body in self.bodies])
        accelerations = self.compute_accelerations()

        # Aktualizacja pozycji
        for i, body in enumerate(self.bodies):
            body.position += body.velocity * dt + 0.5 * accelerations[i] * dt**2

        new_accelerations = self.compute_accelerations()

        # Aktualizacja prędkości
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * (accelerations[i] + new_accelerations[i]) * dt

    def integrate(self, dt, steps):
        positions = []
        for _ in range(steps):
            self.step(dt)
            positions.append([body.position.copy() for body in self.bodies])
        return np.array(positions)

# Cukierek
# body1 = Body(1, [0.517216786720872,0.55610033157918], [0.107632564012758,0.681725256843756])
# body2 = Body(1, [0.002573889407142,0.116484954113653], [-0.534918980283418,-0.854885322576851])
# body3 = Body(1, [-0.20255534902211,-0.731794952123173], [0.427286416269208,0.173160065733631])

# Ładne kółko i 2 ciałka w środku
body1 = Body(1, [0.6661637520772179,-0.081921852656887], [0.84120297540307,0.029746212757039])
body2 = Body(1, [-0.025192663684493022,0.45444857588251897], [0.142642469612081,-0.492315648524683])
body3 = Body(1, [-0.10301329374224,-0.765806200083609], [-0.98384544501151,0.462569435774018])

# Utworzenie układu trzech ciał
system = ThreeBodySystem(body1, body2, body3)

# Symulacja
dt = 0.002  # Umiarkowany krok czasowy dla lepszej równowagi między dokładnością a wydajnością
steps = 20000
positions = system.integrate(dt, steps)

# Animacja
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Adjust to make room for sliders and buttons
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

line1, = ax.plot([], [], 'ro')
line2, = ax.plot([], [], 'go')
line3, = ax.plot([], [], 'bo')
trail1, = ax.plot([], [], 'r-', alpha=0.5)
trail2, = ax.plot([], [], 'g-', alpha=0.5)
trail3, = ax.plot([], [], 'b-', alpha=0.5)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    trail3.set_data([], [])
    return line1, line2, line3, trail1, trail2, trail3

def update(frame):
    line1.set_data([positions[frame, 0, 0]], [positions[frame, 0, 1]])
    line2.set_data([positions[frame, 1, 0]], [positions[frame, 1, 1]])
    line3.set_data([positions[frame, 2, 0]], [positions[frame, 2, 1]])
    trail1.set_data(positions[:frame, 0, 0], positions[:frame, 0, 1])
    trail2.set_data(positions[:frame, 1, 0], positions[:frame, 1, 1])
    trail3.set_data(positions[:frame, 2, 0], positions[:frame, 2, 1])
    return line1, line2, line3, trail1, trail2, trail3


# Button to start/pause the animation
ax_start = plt.axes([0.8, 0.05, 0.1, 0.04])
start_button = Button(ax=ax_start, label='Start/Pause')

anim_running = True


def on_start_button_clicked(event):
    global anim_running
    if anim_running:
        ani.event_source.stop()
    else:
        ani.event_source.start()
    anim_running = not anim_running

start_button.on_clicked(on_start_button_clicked)

ani = FuncAnimation(fig, update, frames=range(steps), init_func=init, interval=1, blit=True)
plt.show()
