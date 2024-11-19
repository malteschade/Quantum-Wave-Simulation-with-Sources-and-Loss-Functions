# -------- Imports --------
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------- Functions --------
# -- Source Functions --
def Gaussian(f, x, y, z, x0=0, y0=0, z0=0):
    """3D Gaussian source."""
    return np.exp(-np.pi**2 * f**2 * ((x - x0)**2 + (y - y0)**2 + (z - z0)**2))

def Ricker(f, x, y, z, x0=0, y0=0, z0=0):
    """Generate a 3D Ricker wavelet."""
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    pi_f_r = np.pi * f * r
    return (1 - 2 * pi_f_r**2) * np.exp(-pi_f_r**2)

# -- Simulation Functions --
def FD(dx, Nx):
    """Finite Difference operator for first derivative. First order accurate. All DBC."""
    return (1 / dx) * sp.diags([-1, 1], [0, -1], shape=(Nx, Nx), format='lil')[:, :-1]

def compute_B(c_model, rho_model, rho_stag_x, rho_stag_y):
    """Compute the material matrices for the 2D acoustic wave equation."""
    # -------- Matrix B (Acoustic) --------
    # Flatten the arrays
    c_model = c_model.flatten()
    rho_model = rho_model.flatten()
    rho_stag_x = rho_stag_x.flatten()
    rho_stag_y = rho_stag_y.flatten()

    # Material matrices for u and v (sqrt)
    B_u_sqrt = sp.diags(1 / np.sqrt(rho_model * c_model ** 2), format='csr')
    B_vx_sqrt = sp.diags(np.sqrt(rho_stag_x), format='csr')
    B_vy_sqrt = sp.diags(np.sqrt(rho_stag_y), format='csr')
    B_sqrt = sp.block_diag([B_u_sqrt, B_vx_sqrt, B_vy_sqrt], format='csr')

    # Material matrices for u and v (sqrt_inv)
    B_u_inv_sqrt = sp.diags(np.sqrt(rho_model * c_model ** 2), format='csr')
    B_vx_inv_sqrt = sp.diags(1 / np.sqrt(rho_stag_x), format='csr')
    B_vy_inv_sqrt = sp.diags(1 / np.sqrt(rho_stag_y), format='csr')
    B_inv_sqrt = sp.block_diag([B_u_inv_sqrt, B_vx_inv_sqrt, B_vy_inv_sqrt], format='csr')
    
    return B_sqrt**2, B_sqrt, B_inv_sqrt**2, B_inv_sqrt

def compute_A(Nx, Ny, dx, dy, bcs):
    """Compute the system matrix for the 2D acoustic wave equation."""
    # -------- Matrix A (Acoustic) --------
    # Finite difference operators
    FD_x = FD(dx, Nx)
    FD_y = FD(dy, Ny)

    # Boundary conditions (Dirichlet or Neumann)
    if bcs['L'] == 'NBC': FD_x[0, :] = 0 
    if bcs['R'] == 'NBC': FD_x[-1, :] = 0
    if bcs['T'] == 'NBC': FD_y[0, :] = 0 
    if bcs['B'] == 'NBC': FD_y[-1, :] = 0

    # Derivative operators
    D_x = sp.kron(sp.eye(Ny), FD_x)
    D_y = sp.kron(FD_y, sp.eye(Nx))

    # Divergence & Gradient operators
    Div = sp.hstack([D_x, D_y], format='csr')
    Grad = -Div.T

    # System matrix A
    A = sp.bmat([
        [None, Div],
        [Grad, None]
    ], format='csr')
    
    return A

def FD_solver_2D(Nx, Ny, dx, dy, c_model, rho_model, rho_stag_x, rho_stag_y, 
                 bcs={'L': 'DBC', 'R': 'DBC', 'T': 'DBC', 'B': 'DBC'}):
    """Compute the evolution Hamiltonian for the 2D acoustic wave equation."""
    # Compute the material matrices
    B, B_sqrt, B_inv, B_inv_sqrt = compute_B(c_model, rho_model, rho_stag_x, rho_stag_y)
    
    # Compute the system matrix
    A = compute_A(Nx, Ny, dx, dy, bcs)
    
    # Compute the Hamiltonian
    H = 1j * B_inv_sqrt @ A @ B_inv_sqrt
    
    return H, A, B, B_sqrt, B_inv, B_inv_sqrt

def FD_solver_2D_quantum(Nx, Ny, dx, dy, c_model, rho_model, rho_stag_x, rho_stag_y,
                         bcs={'L': 'DBC', 'R': 'DBC', 'T': 'DBC', 'B': 'DBC'}):
    """Compute the evolution Hamiltonian for the 2D acoustic wave equation."""
    H, A, B, B_sqrt, B_inv, B_inv_sqrt = FD_solver_2D(Nx, Ny, dx, dy, c_model, rho_model, rho_stag_x, rho_stag_y, bcs)
    # Find next power of 2
    bit = (H.shape[0]-1).bit_length()
    pad = 2**bit - H.shape[0]
    
    # Pad the Hamiltonian with zeros
    H = sp.block_diag([H, sp.csr_matrix((pad, pad))], format='csr')
    A = sp.block_diag([A, sp.csr_matrix((pad, pad))], format='csr')
    B = sp.block_diag([B, sp.csr_matrix((pad, pad))], format='csr')
    B_sqrt = sp.block_diag([B_sqrt, sp.csr_matrix((pad, pad))], format='csr')
    B_inv = sp.block_diag([B_inv, sp.csr_matrix((pad, pad))], format='csr')
    B_inv_sqrt = sp.block_diag([B_inv_sqrt, sp.csr_matrix((pad, pad))], format='csr')
    
    return H, A, B, B_sqrt, B_inv, B_inv_sqrt

def compute_source_2D(S0, t_span, dx, dy, c0, rho0):
    """Compute the source term for the 2D acoustic wave equation."""
    # -- Grid parameters --
    r_x = int((1/dx)*t_span[1] * c0)  # Radius of the source region (x)
    r_y = int((1/dy)*t_span[1] * c0)  # Radius of the source region (y)
    Nx_S, Ny_S = (2*r_x, 2*r_y)       # Number of grid points in the source region

    # -- Wave field definition --
    u0_S = np.zeros((Ny_S, Nx_S))
    v0x_S = np.zeros((Ny_S, (Nx_S-1)))
    v0y_S = np.zeros(((Ny_S-1), Nx_S))
    phi_0_S = np.hstack([u0_S.flatten(), v0x_S.flatten(), v0y_S.flatten()])

    # -- Material properties --
    c_model_S = c0 * np.ones((Ny_S, Nx_S))
    rho_model_S = rho0 * np.ones((Ny_S, Nx_S))
    rho_stag_x_S = rho0 * np.ones((Ny_S, (Nx_S-1)))
    rho_stag_y_S = rho0 * np.ones(((Ny_S-1), Nx_S))

    # -------- Simulation (2D Acoustic) --------
    (_, A_S, _, _, B_inv_S, _) = FD_solver_2D(Nx_S, Ny_S, dx, dy, c_model_S, rho_model_S, rho_stag_x_S, rho_stag_y_S)

    # -------- Classical source simulation --------
    # Integration definition
    def rhs(t, phi):
        wave = B_inv_S @ A_S @ phi
        source = np.zeros_like(phi)
        source[Nx_S//2*Ny_S + Nx_S//2] = S0(t) # Inject source at the center of the source region
        return wave + source

    # Time Integration
    source_initial = solve_ivp(rhs, t_span, phi_0_S, t_eval=t_span, method='DOP853').y.T[-1]
    S_u = source_initial[:Ny_S*Nx_S].reshape((Ny_S, Nx_S))
    S_vx = source_initial[Ny_S*Nx_S:Ny_S*Nx_S + Ny_S*(Nx_S-1)].reshape((Ny_S, Nx_S-1))
    S_vy = source_initial[Ny_S*Nx_S + Ny_S*(Nx_S-1):].reshape((Ny_S-1, Nx_S))
    
    return S_u, S_vx, S_vy

    
# -- Plotting Functions --
def plot_acoustic_2D(phi, Nx, Ny, dx, dy, title='2D Wave Field', subsample=16, scale=2, width=0.003, clim=(-0.15, 0.15)):
    """Plot the 2D wavefield with velocity vectors."""
    # Calculate the number of points for each field
    N_u = Nx * Ny
    N_vx = (Nx - 1) * Ny
    N_vy = Nx * (Ny - 1)
    
    # Split phi into u, vx, vy
    u = phi[:N_u].reshape(Ny, Nx)
    vx = phi[N_u:N_u + N_vx].reshape(Ny, Nx-1)
    vy = phi[N_u + N_vx:].reshape(Ny-1, Nx)
    
    # Create grid coordinates
    x = np.arange(0, Nx*dx, dx)[::subsample]
    y = np.arange(0, Ny*dy, dy)[::subsample]
    X, Y = np.meshgrid(x, y)
    
    # Subsample the velocity components
    vx_sub = vx[::subsample, ::subsample]
    vy_sub = vy[::subsample, ::subsample]
    
    # Initialize the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the wavefield amplitudes
    plt.imshow(u, cmap='RdBu', extent=[0, Nx*dx, 0, Ny*dy], clim=clim, origin='lower')
    plt.colorbar(label='u field')
    
    # Plot the velocity vectors
    plt.quiver(X, Y, vx_sub, vy_sub, scale=scale, width=width)
    
    # Set plot labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks(plt.xticks()[0], plt.xticks()[0] * dx - Nx/2)
    plt.yticks(plt.yticks()[0], plt.yticks()[0] * dy - Ny/2)
    plt.title(title)
    
    # Show the plot
    plt.show()

def plot_maxwells_3D(phi, Nx, Ny, Nz, dx, dy, dz,
                      title='3D Wave Field',
                      subsample=4, scale_E=1.5, scale_B=1.5,
                      width_E=2, width_B=2
                      ):
    """Plot electric and magnetic fields as quiver plots in 3D."""

    # Calculate total number of points
    N_total = Nx * Ny * Nz 
    
    # Split phi into phi_E and phi_B
    phi_E = phi[:3*N_total]
    phi_B = phi[3*N_total:]

    # Split phi_E into Ex, Ey, Ez
    Ex = phi_E[:N_total].reshape(Nz, Ny, Nx)
    Ey = phi_E[N_total:2*N_total].reshape(Nz, Ny, Nx)
    Ez = phi_E[2*N_total:3*N_total].reshape(Nz, Ny, Nx)

    # Split phi_B into Bx, By, Bz
    Bx = phi_B[:(N_total-Ny*Nz)].reshape(Ny, Nz, (Nx-1))
    By = phi_B[(N_total-Ny*Nz):(2*N_total-Ny*Nz-Nx*Nz)].reshape(Nz, (Ny-1), Nx)
    Bz = phi_B[(2*N_total-Ny*Nz-Nx*Nz):].reshape((Nz-1), Ny, Nx)

    # Create grid coordinates
    x = np.linspace(0, (Nx-1)*dx, Nx)
    y = np.linspace(0, (Ny-1)*dy, Ny)
    z = np.linspace(0, (Nz-1)*dz, Nz)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Subsample the grid for quiver plots
    X_sub = X[::subsample, ::subsample, ::subsample]
    Y_sub = Y[::subsample, ::subsample, ::subsample]
    Z_sub = Z[::subsample, ::subsample, ::subsample]

    # Subsample the vector components
    Ex_sub = Ex[::subsample, ::subsample, ::subsample]
    Ey_sub = Ey[::subsample, ::subsample, ::subsample]
    Ez_sub = Ez[::subsample, ::subsample, ::subsample]

    Bx_sub = Bx[::subsample, ::subsample, ::subsample]
    By_sub = By[::subsample, ::subsample, ::subsample]
    Bz_sub = Bz[::subsample, ::subsample, ::subsample]

    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Electric Field Vectors
    ax.quiver(X_sub, Y_sub, Z_sub,
              Ex_sub, Ey_sub, Ez_sub,
              color='r', length=scale_E,
              linewidth=width_E, label='Electric Field')

    # Plot Magnetic Field Vectors
    ax.quiver(X_sub, Y_sub, Z_sub,
              Bx_sub, By_sub, Bz_sub,
              color='b', length=scale_B,
              linewidth=width_B, label='Magnetic Field')

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Correct origin to upper
    ax.invert_zaxis()

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

def plot_field_3D(state):
    """Plot vector field on spherical grid"""
    # Extract dimensions
    _, n_theta, n_phi, n_radii = state.shape

    # Generate spherical coordinates
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)  # Azimuthal angle
    phi = np.linspace(-np.pi/2, np.pi/2, n_phi)  # Polar angle
    radii = np.linspace(0, 1, n_radii)  # Radii from the center outward

    # Create a meshgrid for the spherical coordinates
    Theta, Phi, R = np.meshgrid(theta, phi, radii, indexing='xy')

    # Convert spherical coordinates to cartesian coordinates
    X = R * np.cos(Phi) * np.cos(Theta)
    Y = R * np.cos(Phi) * np.sin(Theta)
    Z = R * np.sin(Phi)

    # Flatten the arrays for plotting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Flatten the vector components
    U_flat = state[0].flatten()
    V_flat = state[1].flatten()
    W_flat = state[2].flatten()

    # Create a Plotly figure
    fig = go.Figure()

    # Add quiver arrows using cones
    fig.add_trace(
        go.Cone(
            x=X_flat,
            y=Y_flat,
            z=Z_flat,
            u=U_flat,
            v=V_flat,
            w=W_flat,
            colorscale='Blues',
            sizemode='absolute',
            sizeref=0.05,  # Adjust sizeref for arrow size
            anchor="tail",
            showlegend=False,  # Disable legend for this trace
            showscale=False     # Disable colorbar
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X-coordinate',
            yaxis_title='Y-coordinate',
            zaxis_title='Z-coordinate',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis = dict(nticks=4, range=[-1,1],),
            yaxis = dict(nticks=4, range=[-1,1],),
            zaxis = dict(nticks=4, range=[-1,1],),
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        scene_camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.12),
        eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        font=dict(size=22),
        width=800,
        height=800,
    )

    return fig
