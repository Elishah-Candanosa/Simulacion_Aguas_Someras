import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "figure.constrained_layout.use": True,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.28,
    "lines.linewidth": 1.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

def flujo_x(U, g):
    h, hu, hv = U
    u = np.where(h > 0, hu / h, 0.0)
    v = np.where(h > 0, hv / h, 0.0)
    return np.array([hu, hu * u + 0.5 * g * h * h, hu * v])

def flujo_y(U, g):
    h, hu, hv = U
    u = np.where(h > 0, hu / h, 0.0)
    v = np.where(h > 0, hv / h, 0.0)
    return np.array([hv, hv * u, hv * v + 0.5 * g * h * h])

def flujo_rusanov_x(UL, UR, g):
    FL = flujo_x(UL, g)
    FR = flujo_x(UR, g)
    hL, huL, hvL = UL
    hR, huR, hvR = UR
    uL = np.where(hL > 0, huL / hL, 0.0)
    vL = np.where(hL > 0, hvL / hL, 0.0)
    uR = np.where(hR > 0, huR / hR, 0.0)
    vR = np.where(hR > 0, hvR / hR, 0.0)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    a = np.maximum(np.sqrt(uL*uL + vL*vL) + cL, np.sqrt(uR*uR + vR*vR) + cR)
    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL)

def flujo_rusanov_y(UL, UR, g):
    GL = flujo_y(UL, g)
    GR = flujo_y(UR, g)
    hL, huL, hvL = UL
    hR, huR, hvR = UR
    uL = np.where(hL > 0, huL / hL, 0.0)
    vL = np.where(hL > 0, hvL / hL, 0.0)
    uR = np.where(hR > 0, huR / hR, 0.0)
    vR = np.where(hR > 0, hvR / hR, 0.0)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    a = np.maximum(np.sqrt(uL*uL + vL*vL) + cL, np.sqrt(uR*uR + vR*vR) + cR)
    return 0.5 * (GL + GR) - 0.5 * a * (UR - UL)

def aplicar_bc(U, bc="reflective"):
    if bc == "reflective":
        U[:, :, 0] = U[:, :, 1]
        U[:, :, -1] = U[:, :, -2]
        U[1, :, 0] *= -1
        U[1, :, -1] *= -1
        U[:, 0, :] = U[:, 1, :]
        U[:, -1, :] = U[:, -2, :]
        U[2, 0, :] *= -1
        U[2, -1, :] *= -1

def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

Lx, Ly = 1.0, 1.0
nx, ny = 170, 170
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

g = 9.81
CFL = 0.33
bc = "reflective"
hmin = 1e-6
h_humedo = 2e-3
rho = 1000.0

yN = (Y - Y.min()) / (Y.max() - Y.min() + 1e-12)
x0 = 0.40 + 0.06*np.sin(2*np.pi*yN) + 0.03*np.sin(6*np.pi*yN)
eta0 = np.where(X < x0, 0.22, 0.06)

R2 = (X - 0.74)**2 + (Y - 0.55)**2
zb = 0.25 * np.exp(-0.5 * R2 / (0.085*0.085))
mascara_islote = zb > 0.03

zbx = np.zeros_like(zb)
zby = np.zeros_like(zb)
zbx[:, 1:-1] = (zb[:, 2:] - zb[:, :-2]) / (2.0 * dx)
zby[1:-1, :] = (zb[2:, :] - zb[:-2, :]) / (2.0 * dy)

h0 = np.maximum(eta0 - zb, hmin)
hu0 = np.zeros_like(h0)
hv0 = np.zeros_like(h0)
U = np.stack([h0, hu0, hv0], axis=0)

U[0] = np.maximum(U[0], hmin)
humedo = U[0] > 2e-3
U[1] = np.where(humedo, U[1], 0.0)
U[2] = np.where(humedo, U[2], 0.0)
U[0][mascara_islote] = hmin
U[1][mascara_islote] = 0.0
U[2][mascara_islote] = 0.0

obs = mascara_islote
fluido = ~obs
hu, hv = U[1], U[2]
ady_r = fluido & np.roll(obs, -1, axis=1)
ady_l = fluido & np.roll(obs,  1, axis=1)
hu[ady_r] = -np.abs(hu[ady_r])
hu[ady_l] =  np.abs(hu[ady_l])
ady_u = fluido & np.roll(obs, -1, axis=0)
ady_d = fluido & np.roll(obs,  1, axis=0)
hv[ady_u] = -np.abs(hv[ady_u])
hv[ady_d] =  np.abs(hv[ady_d])
U[1], U[2] = hu, hv

aplicar_bc(U, bc)

historial_t = [0.0]
historial_masa = [np.sum(U[0]) * dx * dy]

u0 = np.where(U[0] > 1e-4, U[1] / U[0], 0.0)
v0 = np.where(U[0] > 1e-4, U[2] / U[0], 0.0)
K0 = 0.5 * U[0] * (u0**2 + v0**2)
P0 = 0.5 * g * U[0]**2 + g * U[0] * zb
historial_energia = [np.sum(K0 + P0) * dx * dy]

h = U[0]
hu = U[1]
hv = U[2]
u = np.where(h > h_humedo, hu / h, 0.0)
v = np.where(h > h_humedo, hv / h, 0.0)

mask_L = (~mascara_islote[:, :-1]) & mascara_islote[:, 1:]
mask_R = mascara_islote[:, :-1] & (~mascara_islote[:, 1:])
mask_D = (~mascara_islote[:-1, :]) & mascara_islote[1:, :]
mask_U = mascara_islote[:-1, :] & (~mascara_islote[1:, :])

hL = np.where(h[:, :-1] > h_humedo, h[:, :-1], 0.0)
uL = u[:, :-1]
hR = np.where(h[:, 1:] > h_humedo, h[:, 1:], 0.0)
uR = u[:, 1:]
Fx0_h = (np.sum(0.5 * rho * g * hL[mask_L]**2) - np.sum(0.5 * rho * g * hR[mask_R]**2)) * dy
Fx0_d = (np.sum(rho * hL[mask_L] * uL[mask_L]**2) - np.sum(rho * hR[mask_R] * uR[mask_R]**2)) * dy
Fx0 = Fx0_h + Fx0_d

hD = np.where(h[:-1, :] > h_humedo, h[:-1, :], 0.0)
vD = v[:-1, :]
hU = np.where(h[1:, :] > h_humedo, h[1:, :], 0.0)
vU = v[1:, :]
Fy0_h = (np.sum(0.5 * rho * g * hD[mask_D]**2) - np.sum(0.5 * rho * g * hU[mask_U]**2)) * dx
Fy0_d = (np.sum(rho * hD[mask_D] * vD[mask_D]**2) - np.sum(rho * hU[mask_U] * vU[mask_U]**2)) * dx
Fy0 = Fy0_h + Fy0_d

u_dom = np.where(U[0] > 1e-4, U[1] / U[0], 0.0)
v_dom = np.where(U[0] > 1e-4, U[2] / U[0], 0.0)
v_mag = np.sqrt(u_dom**2 + v_dom**2)
c_dom = np.sqrt(g * np.maximum(U[0], 1e-6))
Fr_dom = np.where(U[0] > 1e-3, v_mag / c_dom, 0.0)
Fr_max0 = np.max(Fr_dom)

c = np.sqrt(g * np.where(h > h_humedo, h, np.nan))
Frx_L = np.abs(u[:, :-1]) / c[:, :-1]
Frx_R = np.abs(u[:, 1:]) / c[:, 1:]
Frx_candidatos = np.concatenate([Frx_L[mask_L].ravel(), Frx_R[mask_R].ravel()]) if (np.any(mask_L) or np.any(mask_R)) else np.array([0.0])
Frx_candidatos = Frx_candidatos[np.isfinite(Frx_candidatos)]
Frx_n_max0 = float(np.max(Frx_candidatos)) if Frx_candidatos.size else 0.0

Fry_D = np.abs(v[:-1, :]) / c[:-1, :]
Fry_U = np.abs(v[1:, :]) / c[1:, :]
Fry_candidatos = np.concatenate([Fry_D[mask_D].ravel(), Fry_U[mask_U].ravel()]) if (np.any(mask_D) or np.any(mask_U)) else np.array([0.0])
Fry_candidatos = Fry_candidatos[np.isfinite(Fry_candidatos)]
Fry_n_max0 = float(np.max(Fry_candidatos)) if Fry_candidatos.size else 0.0

S_hu2_x = np.sum(hL[mask_L] * uL[mask_L]**2) + np.sum(hR[mask_R] * uR[mask_R]**2)
S_h2_x = np.sum(hL[mask_L]**2) + np.sum(hR[mask_R]**2)
Frx_eff0 = float(np.sqrt(S_hu2_x / (g * S_h2_x))) if S_h2_x > 0 else 0.0

S_hv2_y = np.sum(hD[mask_D] * vD[mask_D]**2) + np.sum(hU[mask_U] * vU[mask_U]**2)
S_h2_y = np.sum(hD[mask_D]**2) + np.sum(hU[mask_U]**2)
Fry_eff0 = float(np.sqrt(S_hv2_y / (g * S_h2_y))) if S_h2_y > 0 else 0.0

historial_Fx = [Fx0]
historial_Fy = [Fy0]
historial_Fx_h = [Fx0_h]
historial_Fx_d = [Fx0_d]
historial_Fy_h = [Fy0_h]
historial_Fy_d = [Fy0_d]
historial_Fr_max = [Fr_max0]
historial_Frx_n_max = [Frx_n_max0]
historial_Fry_n_max = [Fry_n_max0]
historial_Frx_eff = [Frx_eff0]
historial_Fry_eff = [Fry_eff0]
historial_ratio_x = [abs(Fx0_d) / (abs(Fx0_h) + 1e-30)]
historial_ratio_y = [abs(Fy0_d) / (abs(Fy0_h) + 1e-30)]
historial_ratio_pred_x = [2.0 * Frx_eff0**2]
historial_ratio_pred_y = [2.0 * Fry_eff0**2]

t = 0.0
t_final = 14
pasos_por_frame = 3
mostrar_superficie_libre = True

fig, ax = plt.subplots(figsize=(7.6, 6.2))
ax.set_title("Aguas someras 2D — 2º orden (MUSCL + TVD-RK2)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.grid(True, which="major", linestyle=":", alpha=0.28)
ax.grid(True, which="minor", linestyle=":", alpha=0.14)
ax.set_aspect("equal")
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)

if mostrar_superficie_libre:
    campo0 = U[0] + zb
    etiqueta_cb = "Superficie libre η [m]"
    vmin, vmax = 0.0, 0.25
else:
    campo0 = U[0]
    etiqueta_cb = "Profundidad h [m]"
    vmin, vmax = 0.0, 0.22

im = ax.imshow(campo0, origin="lower", extent=[0, Lx, 0, Ly], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
cb.set_label(etiqueta_cb)

tierra_vis = np.where(mascara_islote, 1.0, np.nan)
ax.imshow(tierra_vis, origin="lower", extent=[0, Lx, 0, Ly], cmap="Greys", alpha=0.55, interpolation="nearest")
ax.contour(X, Y, mascara_islote.astype(float), levels=[0.5], colors="k", linewidths=1.4, alpha=0.9)

hud = ax.text(
    0.02, 0.98, "",
    transform=ax.transAxes,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(facecolor="white", alpha=0.88, edgecolor="0.25", linewidth=0.8, boxstyle="round,pad=0.35")
)

def update(_):
    global U, t
    for _ in range(pasos_por_frame):
        h_local = U[0]
        hu_local = U[1]
        hv_local = U[2]
        u_local = np.where(h_local > 0, hu_local / h_local, 0.0)
        v_local = np.where(h_local > 0, hv_local / h_local, 0.0)
        c_local = np.sqrt(g * np.maximum(h_local, 0.0))
        amax = np.max(np.sqrt(u_local*u_local + v_local*v_local) + c_local)

        dt = CFL * min(dx, dy) / max(amax, 1e-8)
        if t + dt > t_final:
            dt = t_final - t

        U_trab = np.copy(U)
        U_trab[0][mascara_islote] = hmin
        U_trab[1][mascara_islote] = 0.0
        U_trab[2][mascara_islote] = 0.0
        U_trab[0] = np.maximum(U_trab[0], hmin)
        aplicar_bc(U_trab, bc)
        ny_, nx_ = U_trab.shape[1], U_trab.shape[2]

        dU_x = np.diff(U_trab, axis=2)
        pendiente_x = np.zeros_like(U_trab)
        pendiente_x[:, :, 1:-1] = minmod(dU_x[:, :, :-1], dU_x[:, :, 1:])
        UL_x = U_trab[:, :, :-1] + 0.5 * pendiente_x[:, :, :-1]
        UR_x = U_trab[:, :, 1:] - 0.5 * pendiente_x[:, :, 1:]
        Fx = np.zeros((3, ny_, nx_ - 1))
        for i in range(nx_ - 1):
            Fx[:, :, i] = flujo_rusanov_x(UL_x[:, :, i], UR_x[:, :, i], g)
        caras_solidas_x = mascara_islote[:, :-1] | mascara_islote[:, 1:]
        Fx *= (~caras_solidas_x)[None, :, :]

        dU_y = np.diff(U_trab, axis=1)
        pendiente_y = np.zeros_like(U_trab)
        pendiente_y[:, 1:-1, :] = minmod(dU_y[:, :-1, :], dU_y[:, 1:, :])
        UL_y = U_trab[:, :-1, :] + 0.5 * pendiente_y[:, :-1, :]
        UR_y = U_trab[:, 1:, :] - 0.5 * pendiente_y[:, 1:, :]
        Gy = np.zeros((3, ny_ - 1, nx_))
        for j in range(ny_ - 1):
            Gy[:, j, :] = flujo_rusanov_y(UL_y[:, j, :], UR_y[:, j, :], g)
        caras_solidas_y = mascara_islote[:-1, :] | mascara_islote[1:, :]
        Gy *= (~caras_solidas_y)[None, :, :]

        dUdt1 = np.zeros_like(U_trab)
        dUdt1[:, 1:-1, 1:-1] = (
            - (Fx[:, 1:-1, 1:] - Fx[:, 1:-1, :-1]) / dx
            - (Gy[:, 1:, 1:-1] - Gy[:, :-1, 1:-1]) / dy
        )
        h_rhs = U_trab[0]
        dUdt1[1, 1:-1, 1:-1] += -g * h_rhs[1:-1, 1:-1] * zbx[1:-1, 1:-1]
        dUdt1[2, 1:-1, 1:-1] += -g * h_rhs[1:-1, 1:-1] * zby[1:-1, 1:-1]

        U1 = U + dt * dUdt1
        U1[0] = np.maximum(U1[0], hmin)
        humedo1 = U1[0] > 2e-3
        U1[1] = np.where(humedo1, U1[1], 0.0)
        U1[2] = np.where(humedo1, U1[2], 0.0)
        U1[0][mascara_islote] = hmin
        U1[1][mascara_islote] = 0.0
        U1[2][mascara_islote] = 0.0

        obs1 = mascara_islote
        fluido1 = ~obs1
        hu1, hv1 = U1[1], U1[2]
        ady_r1 = fluido1 & np.roll(obs1, -1, axis=1)
        ady_l1 = fluido1 & np.roll(obs1,  1, axis=1)
        hu1[ady_r1] = -np.abs(hu1[ady_r1])
        hu1[ady_l1] =  np.abs(hu1[ady_l1])
        ady_u1 = fluido1 & np.roll(obs1, -1, axis=0)
        ady_d1 = fluido1 & np.roll(obs1,  1, axis=0)
        hv1[ady_u1] = -np.abs(hv1[ady_u1])
        hv1[ady_d1] =  np.abs(hv1[ady_d1])
        U1[1], U1[2] = hu1, hv1
        aplicar_bc(U1, bc)

        U_trab = np.copy(U1)
        U_trab[0][mascara_islote] = hmin
        U_trab[1][mascara_islote] = 0.0
        U_trab[2][mascara_islote] = 0.0
        U_trab[0] = np.maximum(U_trab[0], hmin)
        aplicar_bc(U_trab, bc)

        dU_x = np.diff(U_trab, axis=2)
        pendiente_x = np.zeros_like(U_trab)
        pendiente_x[:, :, 1:-1] = minmod(dU_x[:, :, :-1], dU_x[:, :, 1:])
        UL_x = U_trab[:, :, :-1] + 0.5 * pendiente_x[:, :, :-1]
        UR_x = U_trab[:, :, 1:] - 0.5 * pendiente_x[:, :, 1:]
        Fx = np.zeros((3, ny_, nx_ - 1))
        for i in range(nx_ - 1):
            Fx[:, :, i] = flujo_rusanov_x(UL_x[:, :, i], UR_x[:, :, i], g)
        Fx *= (~caras_solidas_x)[None, :, :]

        dU_y = np.diff(U_trab, axis=1)
        pendiente_y = np.zeros_like(U_trab)
        pendiente_y[:, 1:-1, :] = minmod(dU_y[:, :-1, :], dU_y[:, 1:, :])
        UL_y = U_trab[:, :-1, :] + 0.5 * pendiente_y[:, :-1, :]
        UR_y = U_trab[:, 1:, :] - 0.5 * pendiente_y[:, 1:, :]
        Gy = np.zeros((3, ny_ - 1, nx_))
        for j in range(ny_ - 1):
            Gy[:, j, :] = flujo_rusanov_y(UL_y[:, j, :], UR_y[:, j, :], g)
        Gy *= (~caras_solidas_y)[None, :, :]

        dUdt2 = np.zeros_like(U_trab)
        dUdt2[:, 1:-1, 1:-1] = (
            - (Fx[:, 1:-1, 1:] - Fx[:, 1:-1, :-1]) / dx
            - (Gy[:, 1:, 1:-1] - Gy[:, :-1, 1:-1]) / dy
        )
        h_rhs = U_trab[0]
        dUdt2[1, 1:-1, 1:-1] += -g * h_rhs[1:-1, 1:-1] * zbx[1:-1, 1:-1]
        dUdt2[2, 1:-1, 1:-1] += -g * h_rhs[1:-1, 1:-1] * zby[1:-1, 1:-1]

        U[:] = 0.5 * U + 0.5 * (U1 + dt * dUdt2)
        U[0] = np.maximum(U[0], hmin)
        humedo2 = U[0] > 2e-3
        U[1] = np.where(humedo2, U[1], 0.0)
        U[2] = np.where(humedo2, U[2], 0.0)
        U[0][mascara_islote] = hmin
        U[1][mascara_islote] = 0.0
        U[2][mascara_islote] = 0.0

        obs2 = mascara_islote
        fluido2 = ~obs2
        hu2, hv2 = U[1], U[2]
        ady_r2 = fluido2 & np.roll(obs2, -1, axis=1)
        ady_l2 = fluido2 & np.roll(obs2,  1, axis=1)
        hu2[ady_r2] = -np.abs(hu2[ady_r2])
        hu2[ady_l2] =  np.abs(hu2[ady_l2])
        ady_u2 = fluido2 & np.roll(obs2, -1, axis=0)
        ady_d2 = fluido2 & np.roll(obs2,  1, axis=0)
        hv2[ady_u2] = -np.abs(hv2[ady_u2])
        hv2[ady_d2] =  np.abs(hv2[ady_d2])
        U[1], U[2] = hu2, hv2
        aplicar_bc(U, bc)

        t += dt
        if t >= t_final:
            break

    masa_actual = np.sum(U[0]) * dx * dy
    u_e = np.where(U[0] > 1e-4, U[1] / U[0], 0.0)
    v_e = np.where(U[0] > 1e-4, U[2] / U[0], 0.0)
    K = 0.5 * U[0] * (u_e**2 + v_e**2)
    P = 0.5 * g * U[0]**2 + g * U[0] * zb
    energia_actual = np.sum(K + P) * dx * dy

    h = U[0]
    hu = U[1]
    hv = U[2]
    u = np.where(h > h_humedo, hu / h, 0.0)
    v = np.where(h > h_humedo, hv / h, 0.0)

    mask_L = (~mascara_islote[:, :-1]) & mascara_islote[:, 1:]
    mask_R = mascara_islote[:, :-1] & (~mascara_islote[:, 1:])
    mask_D = (~mascara_islote[:-1, :]) & mascara_islote[1:, :]
    mask_U = mascara_islote[:-1, :] & (~mascara_islote[1:, :])

    hL = np.where(h[:, :-1] > h_humedo, h[:, :-1], 0.0)
    uL = u[:, :-1]
    hR = np.where(h[:, 1:] > h_humedo, h[:, 1:], 0.0)
    uR = u[:, 1:]
    Fx_h = (np.sum(0.5 * rho * g * hL[mask_L]**2) - np.sum(0.5 * rho * g * hR[mask_R]**2)) * dy
    Fx_d = (np.sum(rho * hL[mask_L] * uL[mask_L]**2) - np.sum(rho * hR[mask_R] * uR[mask_R]**2)) * dy
    Fx = Fx_h + Fx_d

    hD = np.where(h[:-1, :] > h_humedo, h[:-1, :], 0.0)
    vD = v[:-1, :]
    hU = np.where(h[1:, :] > h_humedo, h[1:, :], 0.0)
    vU = v[1:, :]
    Fy_h = (np.sum(0.5 * rho * g * hD[mask_D]**2) - np.sum(0.5 * rho * g * hU[mask_U]**2)) * dx
    Fy_d = (np.sum(rho * hD[mask_D] * vD[mask_D]**2) - np.sum(rho * hU[mask_U] * vU[mask_U]**2)) * dx
    Fy = Fy_h + Fy_d

    u_dom = np.where(U[0] > 1e-4, U[1] / U[0], 0.0)
    v_dom = np.where(U[0] > 1e-4, U[2] / U[0], 0.0)
    v_mag = np.sqrt(u_dom**2 + v_dom**2)
    c_dom = np.sqrt(g * np.maximum(U[0], 1e-6))
    Fr_dom = np.where(U[0] > 1e-3, v_mag / c_dom, 0.0)
    Fr_max = np.max(Fr_dom)

    c = np.sqrt(g * np.where(h > h_humedo, h, np.nan))
    Frx_L = np.abs(u[:, :-1]) / c[:, :-1]
    Frx_R = np.abs(u[:, 1:]) / c[:, 1:]
    Frx_candidatos = np.concatenate([Frx_L[mask_L].ravel(), Frx_R[mask_R].ravel()]) if (np.any(mask_L) or np.any(mask_R)) else np.array([0.0])
    Frx_candidatos = Frx_candidatos[np.isfinite(Frx_candidatos)]
    Frx_n_max = float(np.max(Frx_candidatos)) if Frx_candidatos.size else 0.0

    Fry_D = np.abs(v[:-1, :]) / c[:-1, :]
    Fry_U = np.abs(v[1:, :]) / c[1:, :]
    Fry_candidatos = np.concatenate([Fry_D[mask_D].ravel(), Fry_U[mask_U].ravel()]) if (np.any(mask_D) or np.any(mask_U)) else np.array([0.0])
    Fry_candidatos = Fry_candidatos[np.isfinite(Fry_candidatos)]
    Fry_n_max = float(np.max(Fry_candidatos)) if Fry_candidatos.size else 0.0

    S_hu2_x = np.sum(hL[mask_L] * uL[mask_L]**2) + np.sum(hR[mask_R] * uR[mask_R]**2)
    S_h2_x = np.sum(hL[mask_L]**2) + np.sum(hR[mask_R]**2)
    Frx_eff = float(np.sqrt(S_hu2_x / (g * S_h2_x))) if S_h2_x > 0 else 0.0

    S_hv2_y = np.sum(hD[mask_D] * vD[mask_D]**2) + np.sum(hU[mask_U] * vU[mask_U]**2)
    S_h2_y = np.sum(hD[mask_D]**2) + np.sum(hU[mask_U]**2)
    Fry_eff = float(np.sqrt(S_hv2_y / (g * S_h2_y))) if S_h2_y > 0 else 0.0

    ratio_x = abs(Fx_d) / (abs(Fx_h) + 1e-30)
    ratio_y = abs(Fy_d) / (abs(Fy_h) + 1e-30)

    historial_t.append(t)
    historial_masa.append(masa_actual)
    historial_energia.append(energia_actual)
    historial_Fx.append(Fx)
    historial_Fy.append(Fy)
    historial_Fx_h.append(Fx_h)
    historial_Fx_d.append(Fx_d)
    historial_Fy_h.append(Fy_h)
    historial_Fy_d.append(Fy_d)
    historial_Fr_max.append(Fr_max)
    historial_Frx_n_max.append(Frx_n_max)
    historial_Fry_n_max.append(Fry_n_max)
    historial_Frx_eff.append(Frx_eff)
    historial_Fry_eff.append(Fry_eff)
    historial_ratio_x.append(ratio_x)
    historial_ratio_y.append(ratio_y)
    historial_ratio_pred_x.append(2.0 * Frx_eff**2)
    historial_ratio_pred_y.append(2.0 * Fry_eff**2)

    if mostrar_superficie_libre:
        im.set_data(U[0] + zb)
    else:
        im.set_data(U[0])

    error_masa = abs(masa_actual - historial_masa[0]) / historial_masa[0]
    hud.set_text(
        f"t = {t:6.3f} s\n"
        f"Fr máx (dominio) = {Fr_max:.2f}\n"
        f"Fr normal máx cerca islote: x={Frx_n_max:.2f}  y={Fry_n_max:.2f}\n"
        f"Fr efectivo (caras):         x={Frx_eff:.3f}  y={Fry_eff:.3f}\n"
        f"Error relativo de masa = {error_masa:.2e}\n"
        f"Energía total = {energia_actual:.3e} J"
    )
    return im, hud

anim = FuncAnimation(fig, update, interval=20, blit=False)
plt.show()

historial_t_arr = np.array(historial_t)
historial_masa_arr = np.array(historial_masa)
historial_energia_arr = np.array(historial_energia)

fig2, (ax_masa, ax_energia) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
error_masa_arr = np.abs(historial_masa_arr - historial_masa_arr[0]) / historial_masa_arr[0]
ax_masa.plot(historial_t_arr, error_masa_arr, color="0.15")
ax_masa.set_title("Error relativo de masa")
ax_masa.set_xlabel("Tiempo [s]")
ax_masa.set_ylabel(r"$|M(t)-M(0)|/M(0)$")
ax_masa.grid(True, which="major", linestyle=":", alpha=0.28)
ax_masa.grid(True, which="minor", linestyle=":", alpha=0.14)
ax_masa.set_yscale("log")

energia_norm = historial_energia_arr / historial_energia_arr[0]
ax_energia.plot(historial_t_arr, energia_norm, color="tab:red")
ax_energia.set_title("Energía mecánica total (normalizada)")
ax_energia.set_xlabel("Tiempo [s]")
ax_energia.set_ylabel(r"$E(t)/E(0)$")
ax_energia.grid(True, which="major", linestyle=":", alpha=0.28)
ax_energia.grid(True, which="minor", linestyle=":", alpha=0.14)
ax_energia.axhline(1.0, color="0.3", lw=1.0, alpha=0.5)
plt.show()

Fx_arr = np.array(historial_Fx)
Fy_arr = np.array(historial_Fy)
Fx_h_arr = np.array(historial_Fx_h)
Fx_d_arr = np.array(historial_Fx_d)
Fy_h_arr = np.array(historial_Fy_h)
Fy_d_arr = np.array(historial_Fy_d)

fig3, ax_f = plt.subplots(figsize=(9, 4))
ax_f.plot(historial_t_arr, Fx_arr, label="Fuerza en x (total)", color="tab:blue")
ax_f.plot(historial_t_arr, Fy_arr, label="Fuerza en y (total)", color="tab:orange")
ax_f.set_title("Fuerza hidrodinámica sobre el islote")
ax_f.set_xlabel("Tiempo [s]")
ax_f.set_ylabel("Fuerza [N]")
ax_f.grid(True, which="major", linestyle=":", alpha=0.28)
ax_f.grid(True, which="minor", linestyle=":", alpha=0.14)
ax_f.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.25")
plt.show()


Fr_max_arr = np.array(historial_Fr_max)
Frx_n_max_arr = np.array(historial_Frx_n_max)
Fry_n_max_arr = np.array(historial_Fry_n_max)
Frx_eff_arr = np.array(historial_Frx_eff)
Fry_eff_arr = np.array(historial_Fry_eff)

figFr, axFr = plt.subplots(figsize=(9, 4))
axFr.plot(historial_t_arr, Fr_max_arr, label="Fr máximo (dominio)", color="0.1", lw=2.2)
axFr.plot(historial_t_arr, Frx_n_max_arr, label="Fr normal máx cerca islote (x)", color="tab:blue", alpha=0.55)
axFr.plot(historial_t_arr, Fry_n_max_arr, label="Fr normal máx cerca islote (y)", color="tab:orange", alpha=0.55)
axFr.plot(historial_t_arr, Frx_eff_arr, label="Fr efectivo (caras) (x)", color="tab:blue", lw=2.2)
axFr.plot(historial_t_arr, Fry_eff_arr, label="Fr efectivo (caras) (y)", color="tab:orange", lw=2.2)
axFr.set_title("Diagnóstico de Froude")
axFr.set_xlabel("Tiempo [s]")
axFr.set_ylabel("Froude [-]")
axFr.grid(True, which="major", linestyle=":", alpha=0.28)
axFr.grid(True, which="minor", linestyle=":", alpha=0.14)
axFr.legend(ncol=2, loc="upper right", fontsize=13,frameon=True, framealpha=0.9, edgecolor="0.25")
plt.show()

ratio_x_arr = np.array(historial_ratio_x)
ratio_y_arr = np.array(historial_ratio_y)
ratio_pred_x_arr = np.array(historial_ratio_pred_x)
ratio_pred_y_arr = np.array(historial_ratio_pred_y)

figR, (axRx, axRy) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
axRx.plot(historial_t_arr, ratio_x_arr, label="Medido: |Fx_d|/(|Fx_h|+ε)", color="tab:green")
axRx.plot(historial_t_arr, ratio_pred_x_arr, label="Predicho: 2·Frx²", color="tab:red", ls="--")
axRx.set_title("Cociente dinámico/hidrostático en x")
axRx.set_xlabel("Tiempo [s]")
axRx.set_ylabel("Cociente [-]")
axRx.grid(True, which="major", linestyle=":", alpha=0.28)
axRx.grid(True, which="minor", linestyle=":", alpha=0.14)
axRx.legend(loc="best",fontsize=13, frameon=True, framealpha=0.9, edgecolor="0.25")

axRy.plot(historial_t_arr, ratio_y_arr, label="Medido: |Fy_d|/(|Fy_h|+ε)", color="tab:green")
axRy.plot(historial_t_arr, ratio_pred_y_arr, label="Predicho: 2·Fry²", color="tab:red", ls="--")
axRy.set_title("Cociente dinámico/hidrostático en y")
axRy.set_xlabel("Tiempo [s]")
axRy.set_ylabel("Cociente [-]")
axRy.grid(True, which="major", linestyle=":", alpha=0.28)
axRy.grid(True, which="minor", linestyle=":", alpha=0.14)
axRy.legend(loc="best", fontsize=13, frameon=True, framealpha=0.9, edgecolor="0.25")
plt.show()