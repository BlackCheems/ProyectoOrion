"""
Proyecto Orión - Motor GPU (Taichi Lang)
Simulación masiva de N-Cuerpos usando fuerza bruta paralela en GPU.
Autor: Chris (Rubin1)
"""

import taichi as ti
import numpy as np
import os
import time

# --- INICIALIZAR GPU ---
# arch=ti.gpu intentará usar CUDA (NVIDIA) o Vulkan automáticamente
ti.init(arch=ti.gpu) 

# --- CONFIGURACIÓN ---
INPUT_FILE = "data/processed/simulation_input.npy"
OUTPUT_FILE = "data/processed/trajectory_taichi.npy"
G_REAL = 4.30091e-3  # pc (km/s)^2 / Msun
DT = 0.1             # Paso de tiempo (Millones de años)
STEPS = 500          # Cuántos pasos simulamos (Total 500 * 0.1 = 50 Myr para prueba rápida)
SOFTENING = 100.0    # Parsecs (para evitar que la fuerza sea infinita si chocan)

def run_taichi_simulation():
    print("--- INICIANDO MOTOR GPU (TAICHI CUDA) ---")
    
    # 1. Cargar datos
    data = np.load(INPUT_FILE, allow_pickle=True).item()
    masses_np = data['masses'].astype(np.float32)
    pos_np = data['positions'].astype(np.float32) # (N, 3)
    vel_np = data['velocities'].astype(np.float32) # (N, 3)
    
    N = len(masses_np)
    print(f"--> Cargando {N} galaxias en la VRAM de la RTX 3060...")

    # 2. Reservar memoria en la GPU (Taichi Fields)
    # Vector de 3 dimensiones para Posición y Velocidad
    pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
    mass = ti.field(dtype=ti.f32, shape=N)
    
    # Copiar datos de RAM (CPU) a VRAM (GPU)
    pos.from_numpy(pos_np)
    vel.from_numpy(vel_np)
    mass.from_numpy(masses_np)

    # 3. El Kernel Físico (Esto corre en paralelo en miles de hilos)
    @ti.kernel
    def compute_step():
        # Paralelización automática sobre 'i'
        for i in range(N):
            force = ti.Vector([0.0, 0.0, 0.0])
            p_i = pos[i]
            
            # Bucle interno: Sumar fuerza de todas las otras 'j'
            # Aquí está el sudor: 10,000 x 10,000 iteraciones
            for j in range(N):
                if i != j:
                    diff = pos[j] - p_i
                    r = diff.norm()
                    # Gravedad suavizada (Plummer model simplificado)
                    # F = G * m1 * m2 / (r^2 + e^2)
                    r_eff = ti.sqrt(r**2 + SOFTENING**2)
                    factor = G_REAL * mass[j] / (r_eff**3)
                    force += factor * diff
            
            # Integración Semi-Implícita de Euler
            vel[i] += force * DT
            pos[i] += vel[i] * DT

    # 4. Bucle Principal
    history = [] # Guardaremos en RAM para no saturar la VRAM
    
    print(f"--> Comenzando cálculo de fuerza bruta ({N}^2 interacciones por paso)...")
    start_time = time.time()
    
    for s in range(STEPS):
        compute_step() # <--- La magia ocurre aquí
        
        # Sincronizar GPU y guardar snapshot cada 5 pasos para no llenar el disco
        if s % 5 == 0:
            ti.sync() # Esperar a que la GPU termine
            history.append(pos.to_numpy())
            print(f"\rStep {s}/{STEPS} completado", end="")

    end_time = time.time()
    print(f"\n✅ Simulación GPU completada en {end_time - start_time:.2f} segundos.")
    print(f"   Velocidad: {STEPS / (end_time - start_time):.1f} pasos/segundo")
    
    # 5. Guardar
    np.save(OUTPUT_FILE, np.array(history))
    print(f"--> Datos guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    run_taichi_simulation()