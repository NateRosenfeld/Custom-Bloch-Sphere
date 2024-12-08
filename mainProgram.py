import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation  # Ensure this line is included
import numpy as np


# Define a basic quantum state |ψ(t)>
def initQubitState():
    # Initialize a state |ψ(t)> = α|0> + β|1>
    alpha = 1/np.sqrt(2)
    beta = 1/np.sqrt(2)
    state = np.array([alpha, beta])  # Column vector representation
    return state

# Hamiltonian for a simple qubit system (e.g., a spin in a magnetic field)
def hamiltonian():
    # Example: Pauli-X Hamiltonian for a single qubit
    hamiltonian = np.array([[0, 1], [1, 0]])  # Pauli-X matrix
    return hamiltonian

def plotBlochSphere(theta, phi, ax):
    ax.clear()
    ax.set_aspect('equal')

    # Define the Bloch sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xSphere = np.outer(np.cos(u), np.sin(v))
    ySphere = np.outer(np.sin(u), np.sin(v))
    zSphere = np.outer(np.ones_like(u), np.cos(v))

    # Plot the sphere
    ax.plot_surface(xSphere, ySphere, zSphere, rstride=4, cstride=4, color='b', alpha=0.5)

    # Calculate the vector direction (convert spherical to Cartesian)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Plot the state vector
    ax.quiver(0, 0, 0, x, y, z, length=1, normalize=True, color='r')

    # X-axis
    ax.quiver(0, 0, 0, 1.1, 0, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, -1.1, 0, 0, color='g', arrow_length_ratio=0.1)
    ax.text(1.2, 0, 0, "X", color='g', fontsize=12, ha='center')
    ax.text(-1.2, 0, 0, "-X", color='g', fontsize=12, ha='center')

    # Y-axis
    ax.quiver(0, 0, 0, 0, 1.1, 0, color='orange', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, -1.1, 0, color='orange', arrow_length_ratio=0.1)
    ax.text(0, 1.2, 0, "Y", color='orange', fontsize=12, ha='center')
    ax.text(0, -1.2, 0, "-Y", color='orange', fontsize=12, ha='center')

    # Z-axis (Probability axis) with Dirac notation
    axis_length = 1.1  # Extend slightly beyond the sphere
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='purple', arrow_length_ratio=0.1)
    ax.text(0, 0, axis_length + 0.1, "|1⟩", color='purple', fontsize=12, ha='center')
    ax.quiver(0, 0, 0, 0, 0, -axis_length, color='purple', arrow_length_ratio=0.1)
    ax.text(0, 0, -axis_length - 0.1, "|0⟩", color='purple', fontsize=12, ha='center')

    # Remove external gridlines and ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Set limits and title
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title("Bloch Sphere")


# Function to animate the vector evolution
def animateBlochSphere(startTheta, startPhi, endTheta, endPhi, steps):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate intermediate angles for animation
    thetaValues = np.linspace(startTheta, endTheta, steps)
    phiValues = np.linspace(startPhi, endPhi, steps)

    # Store the initial view angle
    init_elev = 20
    init_azim = 45
    ax.view_init(elev=init_elev, azim=init_azim)

    def update(frame):
        # Get current view angles before clearing
        elev = ax.elev
        azim = ax.azim

        ax.cla()  # Clear the axis
        theta = thetaValues[frame]
        phi = phiValues[frame]
        plotBlochSphere(theta, phi, ax)

        # Restore the view angle
        ax.view_init(elev=elev, azim=azim)
        return ax.get_children()

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=50,
        repeat=True,
        blit=False  # Important for interactive rotation
    )

    plt.show()

def evolveQuantumState(state, hamiltonian, timeStep, totalTimeSteps):
    hBar = 1
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Store states for animation
    states = []
    current_state = state.copy()

    # Precompute all states
    for _ in range(totalTimeSteps):
        U = np.eye(2) - 1j * hamiltonian * timeStep / hBar
        current_state = U @ current_state
        current_state = current_state / np.linalg.norm(current_state)
        states.append(current_state.copy())

    def animate(frame):
        ax.clear()
        current_state = states[frame]
        alpha, beta = current_state
        theta = 2 * np.arccos(np.abs(alpha))
        phi = np.angle(beta / alpha) if alpha != 0 else 0
        plotBlochSphere(theta, phi, ax)

        # Need to return all artists that were created
        artists = ax.get_children()
        return artists

    anim = animation.FuncAnimation(fig, animate, frames=totalTimeSteps, interval=50, blit=False)  # Changed to False

    plt.show()
    return states[-1]


def evolve_between_points(start_theta, start_phi, end_theta, end_phi, steps=100):
    """
    Evolve the quantum state from one point on the Bloch sphere to another.
    Parameters are in radians.
    """

    # Convert spherical coordinates to Cartesian unit vectors
    def spherical_to_cartesian(theta, phi):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z])

    # Calculate start and end vectors
    start_vec = spherical_to_cartesian(start_theta, start_phi)
    end_vec = spherical_to_cartesian(end_theta, end_phi)

    # Calculate rotation axis and angle
    rotation_axis = np.cross(start_vec, end_vec)
    if np.all(rotation_axis == 0):
        rotation_axis = np.array([0, 0, 1])  # Default axis if vectors are parallel
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Calculate rotation angle
    cos_angle = np.clip(np.dot(start_vec, end_vec), -1.0, 1.0)
    rotation_angle = np.arccos(cos_angle)

    # Create time-dependent Hamiltonian components
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    H = (rotation_axis[0] * sigma_x +
         rotation_axis[1] * sigma_y +
         rotation_axis[2] * sigma_z)

    # Scale Hamiltonian by total rotation angle
    H *= rotation_angle / steps

    # Now use animateBlochSphere with these parameters
    animateBlochSphere(start_theta, start_phi, end_theta, end_phi, steps)

def get_user_input():
    print("\nEnter coordinates in radians (π = 3.14159...)")
    print("For example: π/2 = 1.57079, π = 3.14159")

    start_theta = float(input("\nEnter starting theta (0 to π): "))
    start_phi = float(input("Enter starting phi (0 to 2π): "))
    end_theta = float(input("\nEnter ending theta (0 to π): "))
    end_phi = float(input("Enter ending phi (0 to 2π): "))

    return start_theta, start_phi, end_theta, end_phi


if __name__ == "__main__":
    print("Welcome to Bloch Sphere Evolution!")
    print("\nSome common points:")
    print("|0⟩ state: theta = 0, phi = 0")
    print("|1⟩ state: theta = π, phi = 0")
    print("|+⟩ state: theta = π/2, phi = 0")
    print("|-⟩ state: theta = π/2, phi = π")
    print("|+i⟩ state: theta = π/2, phi = π/2")
    print("|-i⟩ state: theta = π/2, phi = 3π/2")

    # Let user choose between manual input or preset example
    choice = input("\nWould you like to (1) enter custom coordinates or (2) see a preset example? Enter 1 or 2: ")

    if choice == "1":
        try:
            coords = get_user_input()
            evolve_between_points(*coords)
        except ValueError as e:
            print("\nError: Please enter valid numbers.")
    else:
        # Preset example: rotate from |0⟩ to |1⟩
        print("\nShowing preset example: |0⟩ to |1⟩")
        start_theta = 0  # |0⟩ state (north pole)
        start_phi = 0
        end_theta = np.pi  # |1⟩ state (south pole)
        end_phi = 0
        evolve_between_points(start_theta, start_phi, end_theta, end_phi)
