import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import spherical_coordinates


# Gravitational constant (arbitrary units for simplicity)
G = 1
mass_star = 100  # Mass of the star (arbitrary units)

# Time step
dt = 0.01
# Number of simulation steps
steps = 1000

# Initial position in spherical coordinates (azimuth, zenith)
azimuth_rad = 0.0
zenith_rad = np.pi / 2 - np.pi/4 # In the xy-plane
r = 5.0  # Radial distance from the star

# Convert spherical coordinates to Cartesian
cx, cy, cz = spherical_coordinates.az_zd_to_cx_cy_cz(azimuth_rad, zenith_rad)
pos = np.array([r * cx, r * cy, r * cz])

# Orbital velocity based on Kepler's Third Law: v = sqrt(GM/r)
velocity_magnitude = np.sqrt(G * mass_star / r)

# Set initial velocity perpendicular to the radial direction
velocity = np.array([0.0, velocity_magnitude, 0.0])

# Arrays to store the positions of the planet over time for plotting
positions = []

# Simulation loop
for _ in range(steps):
    # Calculate the gravitational force in Cartesian coordinates
    r = np.linalg.norm(pos)
    force = -G * mass_star * pos / r**3  # Gravitational force
    
    # Update velocity and position
    velocity += force * dt
    pos += velocity * dt

    # Store the position for plotting
    positions.append(pos.copy())

# Convert position data to numpy array for easy plotting
positions = np.array(positions)

# Create figure and 3D axis for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the star (at the origin)
star = ax.scatter(0, 0, 0, color='yellow', s=500, label='Star')

# Plot initial position of the planet
planet, = ax.plot([], [], [], 'bo', markersize=8)  # Planet

# Set up the axes limits
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_zlim([-6, 6])

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Animation function to update the position of the planet
def update(num):
    # Set new data for the planet at each time step
    planet.set_data([positions[num, 0]], [positions[num, 1]])  # Update X and Y as sequences
    planet.set_3d_properties([positions[num, 2]])  # Update Z as a sequence
    return planet,

# Create the animation object
ani = FuncAnimation(fig, update, frames=steps, interval=10, blit=True)

# Show the animation
plt.show()


# Initial position in spherical coordinates (azimuth, zenith)
azimuth_rad = 0.0
zenith_rad = np.pi / 2  # In the xy-plane
r = 5.0  # Radial distance from the star

# Convert spherical coordinates to Cartesian
cx, cy, cz = spherical_coordinates.az_zd_to_cx_cy_cz(azimuth_rad, zenith_rad)
pos = np.array([r * cx, r * cy, r * cz])

# Orbital velocity based on Kepler's Third Law: v = sqrt(GM/r)
velocity_magnitude = np.sqrt(G * mass_star / r)

# Set initial velocity perpendicular to the radial direction
velocity = np.array([0.0, velocity_magnitude, 0.0])

# Simulation loop and printing X, Y, Z, and phi data
print(f"{'Step':<10}{'X':<15}{'Y':<15}{'Z':<15}{'Phi (Azimuth)':<15}")
for step in range(steps):
    # Calculate the gravitational force in Cartesian coordinates
    r = np.linalg.norm(pos)
    force = -G * mass_star * pos / r**3  # Gravitational force
    
    # Update velocity and position
    velocity += force * dt
    pos += velocity * dt

    # Convert current Cartesian position to spherical coordinates
    az, zd = spherical_coordinates.cx_cy_cz_to_az_zd(pos[0], pos[1], pos[2])
    
    # Print the X, Y, Z, and azimuth (phi) data
    print(f"{step:<10}{pos[0]:<15.6f}{pos[1]:<15.6f}{pos[2]:<15.6f}{az:<15.6f}")
    
    if step == 1000:
        break



# Time step
dt = 0.01
# Number of simulation steps
steps = 1000

# Initial position in spherical coordinates (azimuth, zenith)
azimuth_rad = 0.0
zenith_rad = np.pi / 4  # In the xy-plane
r = 5.0  # Radial distance from the star

# Convert spherical coordinates to Cartesian
cx, cy, cz = spherical_coordinates.az_zd_to_cx_cy_cz(azimuth_rad, zenith_rad)
pos = np.array([r * cx, r * cy, r * cz])

# Orbital velocity based on Kepler's Third Law: v = sqrt(GM/r)
velocity_magnitude = np.sqrt(G * mass_star / r)

# Set initial velocity perpendicular to the radial direction
velocity = np.array([0.0, velocity_magnitude, 0.0])

# Arrays to store the positions and azimuth values (phi) over time for plotting
positions = []
phi_values = []

# Simulation loop and storing phi values
for step in range(steps):
    # Calculate the gravitational force in Cartesian coordinates
    r = np.linalg.norm(pos)
    force = -G * mass_star * pos / r**3  # Gravitational force
    
    # Update velocity and position
    velocity += force * dt
    pos += velocity * dt

    # Convert current Cartesian position to spherical coordinates
    az, zd = spherical_coordinates.cx_cy_cz_to_az_zd(pos[0], pos[1], pos[2])
    
    # Store the azimuth (phi) value
    phi_values.append(az)
    
    # Store the position for reference (not used in this specific plot)
    positions.append(pos.copy())

# Time steps for the x-axis of the plot
time_steps = np.arange(steps) * dt

# Plot the phi (azimuth angle) over time
plt.figure(figsize=(8, 6))
plt.plot(time_steps, phi_values, label="Azimuth (phi) over time")
plt.xlabel('Time (s)')
plt.ylabel('Azimuth (phi) [radians]')
plt.title('Azimuth (phi) vs Time for Orbital Simulation')
plt.legend()
plt.grid(True)
plt.show()

