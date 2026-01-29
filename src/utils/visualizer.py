"""
Proyecto Orión - Visualizador de Diagnóstico
Permite inspeccionar las condiciones iniciales generadas para Chimera.
Autor: Chris (Rubin1)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ruta al archivo generado
DATA_PATH = "data/processed/simulation_input.npy"

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: No encuentro el archivo {DATA_PATH}")
        print("   Ejecuta primero: python3 src/chimera/initial_conditions.py")
        exit()
    
    # Cargamos el diccionario (allow_pickle=True es necesario para dicts)
    data = np.load(DATA_PATH, allow_pickle=True).item()
    return data

def plot_chimera_3d(data):
    pos = data['positions'] # (N, 3)
    vel = data['velocities'] # (N, 3)
    masses = data['masses']
    z = data['redshift']
    
    # Convertir a MegaParsecs para que la gráfica sea legible
    pos_mpc = pos / 1e6 
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Título con datos físicos
    ax.set_title(f"Universo Temprano (z={z}) - {len(pos)} Galaxias\nSimulación 'Chimera' - Setup Inicial")
    
    # Usamos la magnitud de la velocidad para el color (Efecto Doppler visual)
    vel_mag = np.linalg.norm(vel, axis=1)
    
    # Scatter Plot
    # s=masses... ajustamos el tamaño del punto según la masa (logarítmico para no tapar todo)
    sizes = np.log10(masses) * 2 
    
    img = ax.scatter(pos_mpc[:,0], pos_mpc[:,1], pos_mpc[:,2], 
                     c=vel_mag, cmap='plasma', s=sizes, alpha=0.8)
    
    # Etiquetas
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    ax.set_zlabel('Z [Mpc]')
    
    # Barra de color (Velocidad)
    cbar = fig.colorbar(img, ax=ax, shrink=0.5)
    cbar.set_label('Velocidad Total (km/s)')
    
    print("📊 Generando visualización 3D...")
    print("   Tip: Usa el mouse para rotar el cubo y buscar los 'clusters'.")
    plt.show()

if __name__ == "__main__":
    sim_data = load_data()
    plot_chimera_3d(sim_data)