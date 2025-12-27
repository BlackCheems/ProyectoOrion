import torch
import matplotlib.pyplot as plt
import time

# --- PARÁMETROS DEL MOTOR DE FÍSICA (Ajustados para estabilidad) ---
N_BODIES = 10000 
G = 1.0           
DT = 0.005        # Paso de tiempo más fino (evita errores de cálculo)
STEPS = 1000      # Más pasos para compensar el DT pequeño
CENTER_MASS = 100.0 # Masa reducida para evitar catapultas inmediatas

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🌌 Simulando {N_BODIES} cuerpos en: {device}")

# --- INICIALIZACIÓN: DISCO DE ACRECIÓN ---
# En lugar de una esfera aleatoria, creamos un disco (anillo)
# Radio entre 2.0 y 8.0 unidades
r = torch.rand(N_BODIES, device=device) * 6.0 + 2.0
theta = torch.rand(N_BODIES, device=device) * 2 * 3.14159

# Convertir polares a cartesianas (x, y)
positions = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=1)

# --- VELOCIDAD ORBITAL ---
# Calculamos la velocidad necesaria para mantener una órbita circular: v = sqrt(GM/r)
v_mag = (G * CENTER_MASS / r).sqrt()

# Vector de velocidad tangente a la órbita (-y, x)
velocities = torch.stack((-v_mag * torch.sin(theta), v_mag * torch.cos(theta)), dim=1)

# Añadimos un poco de "ruido" (caos) para que no sea un círculo perfecto
velocities += torch.randn(N_BODIES, 2, device=device) * 0.1

print("🚀 Iniciando integración orbital...")
start = time.time()

# --- BUCLE PRINCIPAL (Integrador Simpléctico Básico) ---
for i in range(STEPS):
    dist_sq = positions.pow(2).sum(1, keepdim=True) 
    dist = dist_sq.sqrt()
    
    # Gravedad con "Softening" (evita división por cero si chocan)
    force_mag = (G * CENTER_MASS) / (dist_sq + 0.5) 
    acceleration = -positions * (force_mag / dist)
    
    velocities += acceleration * DT
    positions += velocities * DT

torch.cuda.synchronize()
end = time.time()

print(f"✅ Simulación completada en {end - start:.4f}s")

# --- RENDERIZADO ---
positions_cpu = positions.cpu().numpy()

plt.figure(figsize=(10, 10), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

# Puntos más pequeños y transparentes para ver la densidad del gas/polvo
plt.scatter(positions_cpu[:, 0], positions_cpu[:, 1], s=0.2, c='cyan', alpha=0.5)
plt.scatter([0], [0], s=50, c='white', marker='*') # La estrella central

plt.title(f"Nebulosa de Orión: {N_BODIES} Enanas Marrones (Estable)", color='white')
plt.xlim(-12, 12) # Zoom out para ver todo el disco
plt.ylim(-12, 12)
plt.axis('off')

output_file = "orion_v2.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"📸 Imagen generada: {output_file}")