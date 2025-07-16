import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


# Coefficicents

mu = 0.1   # Viscosity -- Fluid Diffusivity

k = 0.2     # Thermal Diffusivity

rho = 2     # Density

g = 10      # Acceleration due to gravity

beta = 2    # (Fictional) Convection force parameter


# -----------------------------------------------------------------------------
# Grid Setup

Lx = 1
Nx = 25
dx = Lx/Nx

x_data = np.linspace( 0, Lx, Nx+1 )

Ly = 1
Ny = 25
dy = Ly/Ny

y_data = np.linspace( 0, Ly, Ny+1 )

X,Y = np.meshgrid( x_data, y_data, indexing='ij' )

# Time step
dt = 5e-3


# -----------------------------------------------------------------------------
# **


def rect_smooth(x, q, L):
    if 0 <= x <= q:
        return 1/q**4 * x**2 * ( x - 2*q )**2
    elif q < x < L-q:
        return 1
    elif L-q <= x <= L:
        return 1/q**4 * ( L-x )**2 * ( L-x - 2*q )**2
    else:
        return 0


def trap_fun(x, q, L):
    s = L*q
    if 0 <= x <= s:
        return x/s
    elif s < x < L-s:
        return 1
    elif L-s <= x <= L:
        return (L-x)/s
    else:
        return 0


def decay_fun(x, y):
    return trap_fun( x, 0.1, Lx ) * trap_fun( y, 0.1, Ly )

decay_mat = np.array( [ [ decay_fun(x,y) for y in y_data ] for x in x_data ] )

decay_arr = decay_mat[1:-1, 1:-1].flatten()




# -----------------------------------------------------------------------------
# Gauss-Siedel Solver
# -----------------------------------------------------------------------------


def GS_solver( A, b, V0, steps ):
    Low = np.tril(A)
    Low_inv = np.linalg.inv( Low )
    Upp = np.triu(A, 1)
    V = V0
    for i in range(steps):
        V = np.matmul( Low_inv, b - np.matmul( Upp, V ) )
    
    return V


# Version for when the matrices L^-1 and U are already known

def GS_solver_const( L_inv, U, b, V0, steps ):
    V = V0
    for i in range(steps):
        V = np.matmul( L_inv, b - np.matmul( U, V ) )
    
    return V



# -----------------------------------------------------------------------------
# Operator Matrices
# -----------------------------------------------------------------------------


Ix = np.identity( Nx-1 )
Iy = np.identity( Ny-1 )

Iprod = np.identity( (Nx-1)*(Ny-1) )


# -----------------------------------------------------------------------------
# Dirichlet Boundary...

ones_x = np.ones(Nx-1)
ones_y = np.ones(Ny-1)

# First order partial derivatives

Px = (  np.diag( ones_x[1:], 1 ) - np.diag( ones_x[1:], -1 )  ) * 0.5/dx
Py = (  np.diag( ones_y[1:], 1 ) - np.diag( ones_y[1:], -1 )  ) * 0.5/dy

Dx = np.kron( Px, Iy )
Dy = np.kron( Ix, Py )

# Second order partial derivatives

P2x = (
           - 2 * np.diag( ones_x )
           + np.diag( ones_x[1:], 1 )
           + np.diag( ones_x[1:], -1 )
       ) / dx**2

P2y = (
           - 2 * np.diag( ones_y )
           + np.diag( ones_y[1:], 1 )
           + np.diag( ones_y[1:], -1 )
       ) / dy**2

D2x = np.kron( P2x, Iy )
D2y = np.kron( Ix, P2y )

# Laplacian Matrix -- Dirichlet
Lap_dir = D2x + D2y


# Advection operator in matrix form
def advect_mat( u, v ):
    u1, v1 = u[..., None], v[..., None]
    return u1 * Dx + v1 * Dy


# -----------------------------------------------------------------------------
# Neumann Boundary ...

# First order partial derivatives

Px_nm = np.zeros_like(Px) + Px
Py_nm = np.zeros_like(Py) + Py

Px_nm[0,0], Px_nm[-1,-1] = -0.5/dx, 0.5/dx
Py_nm[0,0], Py_nm[-1,-1] = -0.5/dy, 0.5/dy

Dx_nm = np.kron( Px_nm, Iy )
Dy_nm = np.kron( Ix, Py_nm )


# Second order partial derivatives

P2x_nm = np.zeros_like(P2x) + P2x
P2x_nm[0,0] = P2x_nm[-1,-1] = -1/dx**2

P2y_nm = np.zeros_like(P2y) + P2y
P2y_nm[0,0] = P2y_nm[-1,-1] = -1/dy**2

# Laplacian matrix -- Neumann
Lap_nm = np.kron( Ix, P2y_nm ) + np.kron( P2x_nm, Iy )




# -----------------------------------------------------------------------------
# Main step -- pre-projection onto divergence free vector fields
# -----------------------------------------------------------------------------


# Dirichlet...

def sys_mat_dir( u, v, dt, k, rho):
    mat0 = - advect_mat(u, v) + k/rho * Lap_dir
    return Iprod - dt * mat0


def Flow_step_dir( u, v, f, S, dt, k, rho ):
    mat = sys_mat_dir( u, v, dt, k, rho )
    mat_inv = np.linalg.inv( mat )
    vec = f + dt * S
    
    return np.matmul( mat_inv, vec )


# Neumann...

def sys_mat_nm( u, v, dt, k, rho ):
    mat0 = -advect_mat(u, v) + k/rho * Lap_nm
    return Iprod - dt * mat0


def Flow_step_nm( u, v, f, S, dt, k, rho ):
    mat = sys_mat_nm( u, v, dt, k, rho )
    mat_inv = np.linalg.inv( mat )
    vec = f + dt * S
    
    return np.matmul( mat_inv, vec )


# -----------------------------------------------------------------------------
# Iterative approach -- for large matrices/vectors
# -----------------------------------------------------------------------------


# GS_steps = 25


# def stepper2d_GS( u, v, f, S, dt, mu, rho ):
#     A = sys_mat( u, v, dt, mu, rho )
#     b = f + dt * S
#     f_out = GS_solver( A, b, f, GS_steps )
#     return f_out


# -----------------------------------------------------------------------------
# Chorin Projection
# -----------------------------------------------------------------------------

h = 1e-3

E = Iprod - h * Lap_nm
E_inv = np.linalg.inv(E)


# -----------------------------------------------------------------------------
# Poisson Equation Solver


# -- Solves D^2 f = S by converging to a steady state solution of
# -- the sourced heat equation: df/dt = D^2 f - S

def Poisson_solver( f0, src, steps ):
    f = f0
    for i in range(steps):
        f = np.matmul( E_inv, f - h*src )
    
    return f


# -- Extract the divergence free part of the flow velocity...

PS_steps = 25

def Chorin( u, v ):
    div = np.matmul( Dx, u ) + np.matmul( Dy, v )
    p0 = np.zeros_like(u)
    p1 = Poisson_solver( p0, div, PS_steps )
    
    u1 = u - np.matmul( Dx_nm, p1 ) * decay_arr
    v1 = v - np.matmul( Dy_nm, p1 ) * decay_arr
    
    return u1, v1



#------------------------------------------------------------------------------
# Gauss-Seidel Variants
# -----------------------------------------------------------------------------


# E_low = np.tril( E )
# E_upp = np.triu( E, 1 )
# E_low_inv = np.linalg.inv( E_low )

# def Poisson_solver_GS( f0, src, steps ):
#     f = f0
#     for i in range(steps):
#         b = f - h*src
#         f = GS_solver_const( E_low_inv, E_upp, b, f, GS_steps )
    
#     return f


# def Chorin_GS( u, v, dt, rho):
#     div = np.matmul( Dx, u ) + np.matmul( Dy, v )
#     p0 = np.zeros_like(u)
#     p1 = Poisson_solver_GS( p0, div, PS_steps )
    
#     u1 = u - np.matmul( Dx_nm, p1 ) * decay_arr
#     v1 = v - np.matmul( Dy_nm, p1 ) * decay_arr
    
#     return u1, v1



# -----------------------------------------------------------------------------
# Main Fluid Stepper Method
# -----------------------------------------------------------------------------


def fluid_stepper2d( u, v, Q, Sx, Sy, Sq, dt, mu, k, rho ):
    u0 = Flow_step_dir( u, v, u, Sx, dt, mu, rho )
    v0 = Flow_step_dir( u, v, v, Sy, dt, mu, rho )
    Q1 = Flow_step_nm( u, v, Q, Sq, dt, k, rho )
    
    u1, v1 = Chorin( u0, v0 )
    
    return u1, v1, Q1



# def fluid_stepper2d_GS( u, v, Sx, Sy, dt, mu, rho ):
#     u1 = stepper2d_GS( u, v, u, Sx, dt, mu, rho )
#     v1 = stepper2d_GS( u, v, v, Sy, dt, mu, rho )
    
#     return Chorin_GS( u1, v1, dt, rho)




# -----------------------------------------------------------------------------
# Unflattening Methods
# -----------------------------------------------------------------------------

# For converting state arrays back into their matrix form


# Dirichlet...

def unflatten( f ):
    mat0 = np.zeros( [ Nx-1, Ny-1 ] )
    for i in range(Nx-1):
        for j in range(Ny-1):
            mat0[i,j] = f[ j + i * (Ny-1) ]
    
    mat1 = np.zeros_like(X)
    mat1[1:-1, 1:-1] = mat0
    
    return mat1


# Neumann...


def unflatten_nm( f ):
    mat0 = np.zeros( [ Nx-1, Ny-1 ] )
    for i in range(Nx-1):
        for j in range(Ny-1):
            mat0[i,j] = f[ j + i * (Ny-1) ]
    
    mat1 = np.zeros_like(X)
    mat1[1:-1, 1:-1] = mat0
    
    mat1[0], mat1[-1] = mat1[1], mat1[-2]
    mat1[:,0], mat1[:,-1] = mat1[:,1], mat1[:,-2]
    
    return mat1


# -----------------------------------------------------------------------------
# Initial data
# -----------------------------------------------------------------------------


def fun1(x, y):
    f1 = 0.5*( 1 - np.cos( 2*np.pi * x / Lx ) )
    f2 = 0.5*( 1 - np.cos( 2*np.pi * y / Ly ) )
    return 1 + f1 * f2


def env_x( x ):
    return rect_smooth( x, 0.1*Lx, Lx/2 )


def env_y( y ):
    return rect_smooth( y, 0.1*Ly, Ly/2 )


wrap1 = np.array( [ [ env_x( x-Lx/4 ) * env_y(y) for y in y_data ]
                   for x in x_data ] )



# Flow Velocity...
u0 = v0 = np.zeros_like(X)
u,v = u0[1:-1, 1:-1].flatten(), v0[1:-1, 1:-1].flatten()

# Heat...
# Q0 = fun1(X,Y)
Q = np.ones( (Nx-1)*(Ny-1) ).flatten()


# Sources...
Sx, Sy = np.zeros_like(Q), - g * (  1 + beta/rho * np.matmul( Dy_nm, Q )  )

Sq = 2.5 * wrap1[1:-1, 1:-1].flatten()



maxstep = 200


# -----------------------------------------------------------------------------
# Plot Setup
# -----------------------------------------------------------------------------


plot_title = ( 'Fluid Velocity Plot: rho='+str(rho)
              + ', mu=' + str(mu)
              + ', k =' + str(k)
              + ', t_max =' + str( maxstep * dt )  )

fig = plt.figure( figsize=[10,8], dpi=80 )
fig.suptitle( plot_title )

ax = fig.add_subplot()

ax.set_xlabel('x')
ax.set_ylabel('y')

minval = 1
maxval = 2

heatmap = ax.pcolormesh( X, Y, np.zeros_like(X), vmin=minval, vmax=maxval )
fig.colorbar( heatmap, ax=ax )

q_plot = ax.quiver( X, Y, np.zeros_like(X), np.zeros_like(X), pivot='middle' )


plt.show()



# -----------------------------------------------------------------------------
# Animation and simulation
# -----------------------------------------------------------------------------


MD = dict( title='', artist='' )
writer = PillowWriter( fps=10, metadata=MD )



with writer.saving( fig, 'Diffusion Plot V2.gif', maxstep ):
    for it in range(maxstep):
        
        heatmap.remove()
        q_plot.remove()
        
        u_mat, v_mat, Q_mat = unflatten(u), unflatten(v), unflatten_nm(Q)
        
        # speed = np.sqrt( u_mat**2 + v_mat**2 )
        
        heatmap = ax.pcolormesh( X, Y, Q_mat, vmin=minval, vmax=maxval )
        q_plot = ax.quiver( X, Y, u_mat, v_mat, pivot='middle' )
        
        Sy = - g * ( 1 + beta/rho * np.matmul( Dy_nm, Q ) )
        
        u,v,Q = fluid_stepper2d( u, v, Q, Sx, Sy, Sq, dt, mu, k, rho )
        
        writer.grab_frame()








# Testing

# for it in range(maxstep):
#     u,v,Q = fluid_stepper2d( u, v, Q, Sx, Sy, Sq, dt, mu, k, rho )


# heatmap.remove()
# q_plot.remove()

# u_mat, v_mat = unflatten( u ), unflatten( v )

# speed = np.sqrt( u_mat**2 + v_mat**2 )

# heatmap = ax.pcolormesh( X, Y, speed, vmin=minval, vmax=maxval )
# q_plot = ax.quiver( X, Y, u_mat, v_mat, pivot='middle' )



