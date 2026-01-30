"""
Proyecto Orión - Módulo Chimera (Initial Conditions Generator)
Generador de escenarios para galaxias densas en alto redshift (z ~ 7).
Autor: Chris (Rubin1)
"""

import numpy as np
import argparse
import os

# --- CONSTANTES FÍSICAS (Unidades: Masas Solares, Parsecs, km/s) ---
G = 4.30091e-3        # pc (km/s)^2 / Msun
H0 = 70.0             # km/s / Mpc (Constante de Hubble hoy)
OMEGA_M = 0.3         # Densidad de materia
OMEGA_L = 0.7         # Energía oscura
REDSHIFT_Z = 7.0      # Universo temprano (hace ~13 mil millones de años)

def get_hubble_parameter(z):
    """Calcula H(z) en el pasado. El universo se expandía más rápido antes."""
    E_z = np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_L)
    return H0 * E_z  # km/s / Mpc

def generate_chimera_scenario(n_galaxies, box_size_mpc, seed):
    np.random.seed(seed)
    
    print(f"--- INICIALIZANDO SIMULACIÓN QUIMERA (Seed: {seed}) ---")
    print(f"Redshift: z={REDSHIFT_Z}")
    print(f"Caja: {box_size_mpc} Mpc^3")
    
    # 1. Definir Masas (Log-Normal Distribution)
    # Basado en datos JWST: Galaxias compactas masivas (10^9 a 10^11 Msun)
    mean_mass = 1e10 
    sigma_mass = 0.5  # Dispersión
    masses = np.random.lognormal(mean=np.log(mean_mass), sigma=sigma_mass, size=n_galaxies)
    masses[0] = 1e13
    # 2. Generar Posiciones "Clustered" (No aleatorias puras)
    # Las galaxias nacen en nidos. Creamos 3 "nidos" principales.
    box_size_pc = box_size_mpc * 1e6
    n_clusters = int(n_galaxies / 10) + 1
    cluster_centers = np.random.rand(n_clusters, 3) * box_size_pc
    
    positions = []
    velocities = []
    
    Hz = get_hubble_parameter(REDSHIFT_Z) # H(z) en km/s / Mpc
    Hz_per_pc = Hz / 1e6 # Convertir a km/s / pc
    
    for i in range(n_galaxies):

        if i == 0:
            # EL REY: En el centro exacto, quieto.
            pos = np.array([box_size_pc/2, box_size_pc/2, box_size_pc/2])
            positions.append(pos)
            velocities.append(np.array([0.0, 0.0, 0.0])) # Velocidad cero
            continue # Saltar al siguiente
        # Elegir un nido aleatorio
        cluster_idx = np.random.randint(0, n_clusters)
        center = cluster_centers[cluster_idx]
        
        # Dispersión alrededor del nido (Gaussian blob ~200 kpc)
        offset = np.random.randn(3) * 200000 
        pos = center + offset
        
        # Condiciones de frontera periódicas (Pac-Man)
        pos = pos % box_size_pc
        positions.append(pos)
        
        # 3. Velocidades: Flujo de Hubble + Velocidad Peculiar
        # Flujo de Hubble: V = H(z) * distancia (expansion del universo)
        # Nota: Calculamos la velocidad relativa al centro de la caja para simplificar
        dist_from_center = pos - (box_size_pc / 2)
        v_hubble = dist_from_center * Hz_per_pc
        
        # Velocidad Peculiar (Movimiento propio, caída hacia el nido)
        # Las galaxias en cúmulos se mueven rápido (~500 km/s)
        v_peculiar = np.random.randn(3) * 100
        
        total_vel = v_hubble + v_peculiar
        velocities.append(total_vel)

    return masses, np.array(positions), np.array(velocities)

def save_data(masses, pos, vel, filename="simulation_input.npy"):
    # Guardamos en formato estructurado para que Rebound y Taichi lo entiendan
    data = {
        "redshift": REDSHIFT_Z,
        "masses": masses,
        "positions": pos,
        "velocities": vel
    }
    
    # Crear carpeta data si no existe
    os.makedirs("data/processed", exist_ok=True)
    filepath = os.path.join("data/processed", filename)
    
    np.save(filepath, data)
    print(f"✅ Datos guardados exitosamente en: {filepath}")

if __name__ == "__main__":
    # Interfaz de Línea de Comandos (CLI)
    parser = argparse.ArgumentParser(description="Generador de Condiciones Iniciales - Proyecto Chimera")
    parser.add_argument("--n", type=int, default=100, help="Número de galaxias")
    parser.add_argument("--box", type=float, default=5.0, help="Tamaño de la caja en Mpc")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    
    args = parser.parse_args()
    
    m, p, v = generate_chimera_scenario(args.n, args.box, args.seed)
    save_data(m, p, v)