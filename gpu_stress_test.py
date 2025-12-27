import torch
import time
import math

# --- CONFIGURACIÓN DE LA MISIÓN ---
N_BODIES = 15000       # 30 mil cuerpos (Carga pesada para la 3060)
N_STEPS = 5000         # Duración de la simulación
G = 1.0                # Gravedad simplificada
SOFTENING = 0.1        # Para evitar divisiones por cero al colisionar

# Verificar si tenemos los propulsores encendidos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Iniciando secuencia de prueba en: {torch.cuda.get_device_name(0)}")
print(f"🌌 Simulando {N_BODIES} cuerpos durante {N_STEPS} pasos de tiempo.")

# --- GENERACIÓN DE DATOS EN VRAM (Tensores) ---
# Posiciones aleatorias en un cubo (distribución normal)
pos = torch.randn(N_BODIES, 3, device=device, dtype=torch.float32)
# Velocidades aleatorias
vel = torch.randn(N_BODIES, 3, device=device, dtype=torch.float32)
# Masas (todas iguales para simplificar)
mass = torch.ones(N_BODIES, 1, device=device, dtype=torch.float32)

# --- MOTOR DE FÍSICA (CUDA KERNEL) ---
def get_acc(pos, mass, G, softening):
    # Truco de álgebra lineal para calcular todas las distancias a la vez (Matriz N x N)
    # x^2 + y^2 + z^2
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    
    # r_ij = pos_j - pos_i
    # Expandimos dimensiones para restar todos contra todos
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    
    # Distancia inversa al cubo (1/r^3) con softening
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3.sqrt_()
    inv_r3.pow_(-3)
    
    # F = G * m * r / r^3
    # La aceleración es la suma de fuerzas de todos los vecinos
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass
    
    return torch.cat((ax, ay, az), dim=1)

# --- BUCLE DE EJECUCIÓN ---
start_time = time.time()
print("⚡ Comienza el estrés térmico...")

for step in range(N_STEPS):
    # 1. Calcular aceleración (La parte pesada)
    acc = get_acc(pos, mass, G, SOFTENING)
    
    # 2. Integración (Verlet o Euler simple)
    vel += acc * 0.01
    pos += vel * 0.01
    
    # Reporte de estado cada 100 pasos
    if step % 100 == 0:
        elapsed = time.time() - start_time
        progress = (step / N_STEPS) * 100
        # Calcular TFLOPS estimados (muy aprox)
        # Operaciones ≈ 20 * N^2 por paso
        ops = 20 * (N_BODIES**2) * step
        tflops = (ops / elapsed) / 1e12 if step > 0 else 0
        
        print(f"Step {step}/{N_STEPS} [{progress:.1f}%] - T: {elapsed:.1f}s - {tflops:.2f} TFLOPS")

total_time = time.time() - start_time
print(f"\n✅ Misión Cumplida.")
print(f"Tiempo Total: {total_time:.2f} segundos ({total_time/60:.2f} minutos).")
print("El sistema es estable. ¡Bienvenido de vuelta, piloto!")