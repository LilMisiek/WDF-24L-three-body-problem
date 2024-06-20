import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import math

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



# algorytm prędkościowy Verleta
    def step(self, dt):
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
choice = int(input("Podaj numer symulacji\n"))

match choice:
    case 1:
        print("Wybrana symulacja : Ładne kółko i 2 ciałka w środku")
        body1 = Body(1, [0.6661637520772179, -0.081921852656887], [0.84120297540307, 0.029746212757039])
        body2 = Body(1, [-0.025192663684493022, 0.45444857588251897], [0.142642469612081, -0.492315648524683])
        body3 = Body(1, [-0.10301329374224, -0.765806200083609], [-0.98384544501151, 0.462569435774018])
        scale = 2
    case 2:
        print("Wybrana symulacja : 3 ciała 1 orbita")
        L = math.sqrt(3)/2
        body1 = Body(1.7, [0, 1], [-1, 0])
        body2 = Body(1.7, [-L, -0.5], [0.5, -L])
        body3 = Body(1.7, [L, -0.5], [0.5, L])
        scale = 2
    case 3:
        print("Wybrana symulacja : 2 ciała na tej samej krzywej")
        body1 = Body(1, [0.486657678894505,0.755041888583519], [-0.182709864466916,0.363013287999004])
        body2 = Body(1, [-0.681737994414464,0.29366023319721], [-0.579074922540872,-0.748157481446087])
        body3 = Body(1, [-0.02259632746864,-0.612645601255358], [0.761784787007641,0.385144193447218])
        scale = 2
    case 4:
        print("Wybrana symulacja : Kwiatek")
        body1 = Body(1, [-0.0039949015,0.0000000000], [0.0000000000,1.1854800959])
        body2 = Body(1, [1.1246615709,0.0000000000], [0.0000000000,-0.2282781803])
        body3 = Body(1, [-1.1206666694,0.0000000000], [0.0000000000,-0.9572019156])
        scale = 2
    case 5:
        print("Wybrana symulacja : Ósemka")
        body1 = Body(1, [-1,0], [0.347113,0.532727])
        body2 = Body(1, [1,0], [0.347113,0.532727])
        body3 = Body(1, [0,0], [-0.694226,-1.065454])
        scale = 2
    case 6:
        # Kwiatuszek
        print("Wybrana symulacja : Kwiatek 2")
        body1 = Body(1, [0.8822391241,0], [0,1.0042424155])
        body2 = Body(1, [-0.6432718586,0], [0,-1.6491842814])
        body3 = Body(1, [-0.2389672654,0], [0,0.6449418659])
        scale = 2

    case 7:
        # gwiazdka w kolkach
        print("Wybrana symulacja : Gwiazdka w kółkach")
        body1 = Body(1, [-1.1889693067,0], [0,0.8042120498])
        body2 = Body(1, [3.8201881837,0], [0,0.0212794833])
        body3 = Body(1, [-2.631218877,0], [0,-0.8254915331])
        scale = 5
    case 8:
        # plusik
        print("Wybrana symulacja : Plusik")
        body1 = Body(1, [-0.1095519101,0], [0,0.9913358338])
        body2 = Body(1, [1.6613533905,0], [0,-0.1569959746])
        body3 = Body(1, [-1.5518014804,0], [0,-0.8343398592])
        scale = 2
    case 9:
        print("Wybrana symulacja : Układ planetarny")
        body1 = Body(100, [0, 0], [0, -0.07])
        body2 = Body(1, [1, 0], [0, 10])
        body3 = Body(0.5, [4, 0], [0, -5])
        scale = 5
    case _:
        print("domyslny case")

# Utworzenie układu trzech ciał
system = ThreeBodySystem(body1, body2, body3)

# Symulacja
dt = 0.005
steps = 20000
positions = system.integrate(dt, steps)

# Animacja
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(-scale, scale)
ax.set_ylim(-scale, scale)
ax.set_aspect('equal')

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


# Przycisk start/pauza
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
