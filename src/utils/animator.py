"""
Proyecto Orión - Animador Universal (Fixed v2)
Soporta visualización de CPU (Rebound) y GPU (Taichi).
Autor: Chris (Rubin1)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse

# Rutas por defecto
FILE_CPU = "data/processed/trajectory_rebound.npy"
FILE_GPU = "data/processed/trajectory_taichi.npy"
META_FILE = "data/processed/simulation_input.npy"

def animate_chimera(mode='gpu'):
    # Seleccionar archivo
    if mode == 'cpu':
        traj_file = FILE_CPU
        title_suffix = "(CPU - Rebound)"
        point_color = 'cyan'
        alpha_val = 0.8
    else:
        traj_file = FILE_GPU
        title_suffix = "(GPU - Taichi)"
        point_color = 'orange'
        alpha_val = 0.3 

    if not os.path.exists(traj_file):
        print(f"❌ No encuentro {traj_file}. Corre el motor {mode} primero.")
        return

    # Cargar datos
    print(f"--> Cargando datos de {mode.upper()}...")
    traj = np.load(traj_file) # (Snapshots, N, 3)
    meta = np.load(META_FILE, allow_pickle=True).item()
    masses = meta['masses']
    
    # Ajuste de seguridad
    if len(masses) != traj.shape[1]:
        print(f"⚠️ Aviso: El input tiene {len(masses)} masas pero la trayectoria tiene {traj.shape[1]} cuerpos.")
        sizes = np.ones(traj.shape[1]) * 2
    else:
        sizes = np.log10(masses) * 0.5 

    # Configurar Figura
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black') 
    ax.set_facecolor('black')
    
    # Estilo oscuro
    ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.grid(False) 
    
    limit = 20.0
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_zlim(0, limit)
    ax.set_title(f"Chimera: {traj.shape[1]} Galaxias {title_suffix}", color='white')

    # --- INICIALIZACIÓN CORREGIDA ---
    # Usamos el frame 0 en lugar de listas vacías
    pos0 = traj[0]
    x0 = pos0[:,0] / 1e6
    y0 = pos0[:,1] / 1e6
    z0 = pos0[:,2] / 1e6
    
    graph = ax.scatter(x0, y0, z0, s=sizes, c=point_color, alpha=alpha_val, edgecolors='none')
    txt_time = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color='white')

    def update(frame):
        pos = traj[frame]
        x = pos[:,0] / 1e6
        y = pos[:,1] / 1e6
        z = pos[:,2] / 1e6
        
        graph._offsets3d = (x, y, z)
        txt_time.set_text(f"Frame: {frame}")
        return graph,

    print(f"🎬 Renderizando {len(traj)} cuadros con {traj.shape[1]} partículas...")
    ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=20, blit=False)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='gpu', choices=['cpu', 'gpu'], help="Elige motor a visualizar")
    args = parser.parse_args()
    
    animate_chimera(mode=args.mode)