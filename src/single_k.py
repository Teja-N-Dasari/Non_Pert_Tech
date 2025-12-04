import numpy as np
import math
import matplotlib.pyplot as plt

e = 1.602176634e-19
hbar = 1.054571817e-34

# Tight-binding parameters
t_eV = 2.7
t_hop = t_eV * e
Delta = (0.5 * e)
a = 2.46e-10

delta_vec = np.array([
    [0.0, -1.0],
    [np.sqrt(3)/2, 0.5],
    [-np.sqrt(3)/2, 0.5]
]) * a


# Hamiltonian and derivatives
#===========================
def f_k(kx, ky):
    s = 0+0j
    for d in delta_vec:
        s += np.exp(1j*(kx*d[0] + ky*d[1]))
    return -t_hop * s

def H_k(kx, ky):
    fk = f_k(kx, ky)
    return np.array([
        [Delta/2, fk],
        [np.conjugate(fk), -Delta/2]
    ], dtype=complex)

def df_dkx(kx, ky):
    s = 0+0j
    for d in delta_vec:
        s += 1j*d[0]*np.exp(1j*(kx*d[0] + ky*d[1]))
    return -t_hop * s

def df_dky(kx, ky):
    s = 0+0j
    for d in delta_vec:
        s += 1j*d[1]*np.exp(1j*(kx*d[0] + ky*d[1]))
    return -t_hop * s

def dH_dkx(kx, ky):
    df = df_dkx(kx, ky)
    return np.array([[0, df],[np.conjugate(df), 0]], complex)

def dH_dky(kx, ky):
    df = df_dky(kx, ky)
    return np.array([[0, df],[np.conjugate(df), 0]], complex)


# Eigenvalues + eigenvectors
#=============================
def eig_k(kx, ky):
    H = H_k(kx, ky)
    vals, vecs = np.linalg.eigh(H)
    return vals, vecs   # vals[0] -> valance and vals[1] -> conduction


# Interband dipole via covariant derivative
#====================================
def dipole_interband(kx, ky):
    vals, U = eig_k(kx, ky)
    Ev, Ec = vals[0], vals[1]
    if abs(Ec-Ev) < 1e-24:
        return np.zeros(2, dtype=complex)

    u_v = U[:,0]
    u_c = U[:,1]

    vx_op = dH_dkx(kx,ky)/hbar
    vy_op = dH_dky(kx,ky)/hbar

    vcv_x = np.vdot(u_c, vx_op @ u_v)
    vcv_y = np.vdot(u_c, vy_op @ u_v)

    d_cv_x = 1j*hbar*vcv_x/(Ev-Ec)
    d_cv_y = 1j*hbar*vcv_y/(Ev-Ec)

    return np.array([d_cv_x, d_cv_y])


# Intraband dipoles 
#======================
def dipole_intraband(kx, ky, dk=1e7):
    _, U0 = eig_k(kx, ky)
    _, Ux = eig_k(kx+dk, ky)
    _, Uy = eig_k(kx, ky+dk)

    phi_vx = np.angle(np.vdot(U0[:,0], Ux[:,0]))
    phi_vy = np.angle(np.vdot(U0[:,0], Uy[:,0]))
    phi_cx = np.angle(np.vdot(U0[:,1], Ux[:,1]))
    phi_cy = np.angle(np.vdot(U0[:,1], Uy[:,1]))

    d_vv = np.array([phi_vx/dk, phi_vy/dk])
    d_cc = np.array([phi_cx/dk, phi_cy/dk])
    return d_vv, d_cc


# Velocity matrix v_mn
# =======================
def vel_matrix(kx, ky):
    vals, U = eig_k(kx, ky)
    vx = dH_dkx(kx,ky)/hbar
    vy = dH_dky(kx,ky)/hbar

    vmat_x = np.zeros((2,2), dtype=complex)
    vmat_y = np.zeros((2,2), dtype=complex)

    for m in range(2):
        for n in range(2):
            vmat_x[m,n] = np.vdot(U[:,m], vx @ U[:,n])
            vmat_y[m,n] = np.vdot(U[:,m], vy @ U[:,n])

    return vmat_x, vmat_y


# Laser pulse (analytic A(t), E(t))
# ====================================
def A_and_E(t, A0, omega, R, theta, tau, t0):
    sigma = tau/(2*np.sqrt(2*np.log(2)))
    f = np.exp(-(t-t0)**2/(2*sigma**2))

    Ax = A0*f*np.cos(omega*t)
    Ay = A0*f*R*np.cos(2*omega*t)*np.sin(theta)

    df = -(t-t0)/(sigma**2)*f

    dAx = A0*(df*np.cos(omega*t) - omega*f*np.sin(omega*t))
    dAy = A0*R*(df*np.cos(2*omega*t) - 2*omega*f*np.sin(2*omega*t))*np.sin(theta)

    Ex = -dAx
    Ey = -dAy
    return np.array([Ax,Ay]), np.array([Ex,Ey])


# SBE 
#=============================
def drho_dt(P, Q, Evec, dcv, dcc, dvv, gap, T2):
    Pbar = 1-P
    rho_vc = np.conjugate(Q)

    dP = (1j*np.dot(Evec, dcv*rho_vc - np.conjugate(dcv)*Q)).real

    diag_diff = dcc - dvv
    term = dcv*(Pbar-P) + diag_diff*Q

    dQ = 1j*np.dot(Evec, term) - Q/T2 - 1j*(gap/hbar)*Q

    return dP, dQ

#================= Propagation ==============================

# Choose a SINGLE k-point 
#=============================
kx0 = 0.1e10     
ky0 = 0.1e10

# Laser parameters
#============================
wavelength = 3.2e-6
omega = 2*np.pi*3e8 / wavelength
E0 = 5e8
A0 = E0/omega
R = 0.5
theta = np.pi/2
tau = 100e-15
t0 = 3*tau

# Time parameters
#=========================
dt = 0.015e-15
tmax = 2*t0 + 200e-15
Nt = int(tmax/dt)
time = np.arange(Nt)*dt

# SBE parameters
# =========================
T2 = 1.5e-15

# Initialize density matrix
#=================================
P = 0.0                      # rho_cc
Q = 0.0 + 0.0j               # rho_cv

P_arr = np.zeros(Nt)
Q_arr = np.zeros(Nt, dtype=complex)

# RK4 helper for single k
#============================
def rk4_single_step(P, Q, t):

    # k shift kt 
    A_vec, E_vec = A_and_E(t, A0, omega, R, theta, tau, t0)
    kxt = kx0 + (e/hbar)*A_vec[0]
    kyt = ky0 + (e/hbar)*A_vec[1]

    vals, _ = eig_k(kxt, kyt)
    Ev, Ec = vals[0], vals[1]
    gap = Ec - Ev

    dcv = dipole_interband(kxt,kyt)
    dvv, dcc = dipole_intraband(kxt,kyt)

    def rhs(P_, Q_, t_):
        A_,E_ = A_and_E(t_, A0, omega, R, theta, tau, t0)
        kxt_ = kx0 + (e/hbar)*A_[0]
        kyt_ = ky0 + (e/hbar)*A_[1]

        vals_,_ = eig_k(kxt_,kyt_)
        Ev_,Ec_ = vals_[0], vals_[1]
        gap_ = Ec_-Ev_

        dcv_ = dipole_interband(kxt_,kyt_)
        dvv_, dcc_ = dipole_intraband(kxt_,kyt_)

        dP,dQ = drho_dt(P_, Q_, E_, dcv_, dcc_, dvv_, gap_, T2)
        return dP,dQ

    # RK4 substeps
    k1P, k1Q = rhs(P,Q,t)
    k2P, k2Q = rhs(P+0.5*dt*k1P, Q+0.5*dt*k1Q, t+0.5*dt)
    k3P, k3Q = rhs(P+0.5*dt*k2P, Q+0.5*dt*k2Q, t+0.5*dt)
    k4P, k4Q = rhs(P+dt*k3P, Q+dt*k3Q, t+dt)

    P_new = P + (dt/6)*(k1P + 2*k2P + 2*k3P + k4P)
    Q_new = Q + (dt/6)*(k1Q + 2*k2Q + 2*k3Q + k4Q)

    # Enforce positivity
    if P_new < 0: P_new=0
    if P_new > 1: P_new=1
    max_coh = np.sqrt(P_new*(1-P_new))
    if abs(Q_new) > max_coh:
        if abs(Q_new)>1e-18:
            Q_new = Q_new*(max_coh/abs(Q_new))
        else:
            Q_new = 0.0

    return P_new, Q_new


# Time loop
#===========================
for it,t in enumerate(time):
    P_arr[it] = P
    Q_arr[it] = Q
    P,Q = rk4_single_step(P,Q,t)


# Plot population & coherence
# =====================================
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot(time*1e15, P_arr)
plt.xlabel("t (fs)")
plt.ylabel("rho_cc")

plt.subplot(1,2,2)
plt.plot(time*1e15, np.abs(Q_arr))
plt.xlabel("t (fs)")
plt.ylabel("|rho_cv|")

plt.suptitle(f"Single-k Evolution at k=({kx0:.2e},{ky0:.2e})")
plt.tight_layout()
plt.show()


# ----------------- Plotting helper functions -----------------
def plot_eigen_energies_kx(kx_vals, ky=0.0, ax=None):

    kx_vals = np.array(kx_vals)
    E = np.zeros((2, kx_vals.size))
    for i,kx in enumerate(kx_vals):
        vals, _ = eig_k(kx, ky)
        E[:, i] = vals

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.plot(kx_vals, E[0,:]/e, label='Valence')
    ax.plot(kx_vals, E[1,:]/e, label='Conduction')
    ax.set_xlabel('kx (1/m)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(f'Eigen energies vs kx (ky={ky:.2e})')
    ax.legend()
    ax.grid(True)
    return ax


def plot_dipole_interband_kx(kx_vals, ky=0.0):

    kx_vals = np.array(kx_vals)
    dx = np.zeros(kx_vals.size, dtype=complex)
    dy = np.zeros(kx_vals.size, dtype=complex)
    for i,kx in enumerate(kx_vals):
        d = dipole_interband(kx, ky)
        dx[i], dy[i] = d[0], d[1]

    plt.figure()
    plt.plot(kx_vals, dx.real, label='Re d_x')
    plt.plot(kx_vals, dx.imag, label='Im d_x')
    plt.plot(kx_vals, dy.real, '--', label='Re d_y')
    plt.plot(kx_vals, dy.imag, '--', label='Im d_y')
    plt.xlabel('kx (1/m)')
    plt.ylabel('Dipole (m)')
    plt.title(f'Interband dipole vs kx (ky={ky:.2e})')
    plt.legend()
    plt.grid(True)
    return dx, dy


def plot_velocity_elements_kx(kx_vals, ky=0.0):

    kx_vals = np.array(kx_vals)
    n = kx_vals.size
    vx_00 = np.zeros(n, dtype=float)
    vx_11 = np.zeros(n, dtype=float)
    vx_01 = np.zeros(n, dtype=float)

    vy_00 = np.zeros(n, dtype=float)
    vy_11 = np.zeros(n, dtype=float)
    vy_01 = np.zeros(n, dtype=float)

    for i,kx in enumerate(kx_vals):
        vmat_x, vmat_y = vel_matrix(kx, ky)
        vx_00[i] = abs(vmat_x[0,0])
        vx_11[i] = abs(vmat_x[1,1])
        vx_01[i] = abs(vmat_x[0,1])

        vy_00[i] = abs(vmat_y[0,0])
        vy_11[i] = abs(vmat_y[1,1])
        vy_01[i] = abs(vmat_y[0,1])

    plt.figure()
    plt.plot(kx_vals, vx_00, label='|v_x 00|')
    plt.plot(kx_vals, vx_11, label='|v_x 11|')
    plt.plot(kx_vals, vx_01, label='|v_x 01|')
    plt.xlabel('kx (1/m)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Velocity matrix elements (v_x) vs kx (ky={ky:.2e})')
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(kx_vals, vy_00, label='|v_y 00|')
    plt.plot(kx_vals, vy_11, label='|v_y 11|')
    plt.plot(kx_vals, vy_01, label='|v_y 01|')
    plt.xlabel('kx (1/m)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Velocity matrix elements (v_y) vs kx (ky={ky:.2e})')
    plt.legend()
    plt.grid(True)

    return (vx_00, vx_11, vx_01), (vy_00, vy_11, vy_01)


def plot_pulse(time_array=None, A0_in=None, omega_in=None, R_in=None, theta_in=None, tau_in=None, t0_in=None):

    if time_array is None:
        time_array = time
    A0_l = A0 if A0_in is None else A0_in
    omega_l = omega if omega_in is None else omega_in
    R_l = R if R_in is None else R_in
    theta_l = theta if theta_in is None else theta_in
    tau_l = tau if tau_in is None else tau_in
    t0_l = t0 if t0_in is None else t0_in

    A_t = np.zeros((2, len(time_array)))
    E_t = np.zeros((2, len(time_array)))
    for i,t in enumerate(time_array):
        A_vec, E_vec = A_and_E(t, A0_l, omega_l, R_l, theta_l, tau_l, t0_l)
        A_t[:,i] = A_vec
        E_t[:,i] = E_vec

    plt.figure()
    plt.plot(time_array*1e15, A_t[0,:], label='A_x')
    plt.plot(time_array*1e15, A_t[1,:], label='A_y')
    plt.xlabel('t (fs)')
    plt.ylabel('A (V·s/m)')
    plt.title('Vector potential A(t)')
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(time_array*1e15, E_t[0,:], label='E_x')
    plt.plot(time_array*1e15, E_t[1,:], label='E_y')
    plt.xlabel('t (fs)')
    plt.ylabel('E (V/m)')
    plt.title('Electric field E(t)')
    plt.legend()
    plt.grid(True)

    return A_t, E_t

kmax = 4*np.pi/(3*a)
kx_vals = np.linspace(kx0 - kmax, kx0 + kmax, 501)

plot_eigen_energies_kx(kx_vals, ky=ky0)
plot_dipole_interband_kx(kx_vals, ky=ky0)
plot_velocity_elements_kx(kx_vals, ky=ky0)
plot_pulse()

plt.show()

