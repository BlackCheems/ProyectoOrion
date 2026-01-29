"""
Proyecto Orión - Motor CPU (REBOUND Reference)
Simulación de N-Cuerpos de alta precisión para validar colisiones.
Usa el integrador IAS15 (adaptativo).
Autor: Chris (Rubin1)
"""

import rebound
import numpy as np
import os
import time

# --- CONFIGURACIÓN ---
INPUT_FILE = "data/processed/simulation_input.npy"
OUTPUT_FILE = "data/processed/trajectory_rebound.npy"
SIMULATION_TIME = 500e6  # 500 Millones de años
SNAPSHOTS = 100          # Cuántas "fotos" guardamos para la animación

def run_rebound_simulation():
    print("--- INICIANDO MOTOR CPU (REBOUND IAS15) ---")
    
    # 1. Cargar condiciones iniciales
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"No encuentro {INPUT_FILE}")
        
    data = np.load(INPUT_FILE, allow_pickle=True).item()
    masses = data['masses']
    pos = data['positions'] # en Parsecs
    vel = data['velocities'] # en km/s
    
    # 2. Configurar REBOUND
    sim = rebound.Simulation()
    sim.units = ('Msun', 'pc', 'yr') # Masas solares, parsecs, años
    
    # Añadir partículas
    print(f"--> Cargando {len(masses)} galaxias en el integrador...")
    for i in range(len(masses)):
        # Rebound necesita velocidades en pc/yr, no km/s
        # Conversión: 1 km/s ~= 1.02e-6 pc/yr
        km_s_to_pc_yr = 1.022690e-6
        
        sim.add(m=masses[i],
                x=pos[i,0], y=pos[i,1], z=pos[i,2],
                vx=vel[i,0]*km_s_to_pc_yr, 
                vy=vel[i,1]*km_s_to_pc_yr, 
                vz=vel[i,2]*km_s_to_pc_yr)

    # Movemos al centro de masa para estabilidad numérica
    sim.move_to_com()
    
    # Integrador IAS15: Lento pero preciso (maneja encuentros cercanos sin crashear)
    sim.integrator = "ias15"
    
    # 3. Bucle de Tiempo
    times = np.linspace(0, SIMULATION_TIME, SNAPSHOTS)
    history = []
    
    start_time = time.time()
    
    print(f"--> Simulando {SIMULATION_TIME/1e6} Millones de años...")
    for i, t in enumerate(times):
        sim.integrate(t)
        
        # Guardamos posiciones actuales (N, 3)
        positions = np.array([[p.x, p.y, p.z] for p in sim.particles])
        history.append(positions)
        
        # Barra de progreso simple
        prog = (i / SNAPSHOTS) * 100
        print(f"\rProgreso: [{int(prog)}%] - Tiempo simulado: {t/1e6:.1f} Myr", end="")

    end_time = time.time()
    print(f"\n✅ Simulación completada en {end_time - start_time:.2f} segundos.")
    
    # 4. Guardar Resultados
    np.save(OUTPUT_FILE, np.array(history))
    print(f"--> Trayectorias guardadas en {OUTPUT_FILE}")

if __name__ == "__main__":
    run_rebound_simulation()