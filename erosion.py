import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
def terrain_erosion_model(t, Z, kw, ks, rx, ry):
    n = int(np.sqrt(len(Z)))
    Z = Z.reshape((n, n))
    dx = np.gradient(Z, rx, axis=1)
    dy = np.gradient(Z, ry, axis=0)
    water_erosion = kw * (np.abs(dx) + np.abs(dy))
    stability = ks * np.exp(-np.sqrt(dx**2 + dy**2))
    dZ = -water_erosion + stability
    return dZ.flatten()
def plot_terrain(ax, X, Y, Z, title):
    surf=ax.plot_surface(X, Y, Z,cmap='terrain',linewidth=0, antialiased=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title(title)
    return surf
n = 50
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
Z0=(np.exp(-((X-0.3)**2+(Y-0.3)**2)/0.1)+np.exp(-((X-0.7)**2+(Y-0.7)**2)/0.1))
kw = 0.5
ks = 0.1
t_span = (0, 2)
t_eval = np.linspace(0, 2, 20)
solution = solve_ivp(
    terrain_erosion_model, 
    t_span,
    Z0.flatten(),
    t_eval=t_eval,
    args=(kw, ks, x[1]-x[0], y[1]-y[0]),
    method='RK45'
)
figure = plt.figure(figsize=(15, 5))
ax1 = figure.add_subplot(131, projection='3d')
plot_terrain(ax1, X, Y, Z0, 'Initial Terrain')
mid_idx = len(solution.t)//2
ax2 = figure.add_subplot(132, projection='3d')
mid_point = solution.y[:,mid_idx].reshape(n, n)
plot_terrain(ax2, X, Y, mid_point, 'Mid-point Erosion')
ax3 = figure.add_subplot(133, projection='3d')
final_state = solution.y[:,-1].reshape(n, n)
plot_terrain(ax3, X, Y, final_state, 'Final Terrain')
plt.tight_layout()
plt.show()