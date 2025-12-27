import torch
import matplotlib
matplotlib.use('Agg') # Backend sin cabeza (vital para servidores sin monitor)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# --- CONFIGURACIÓN DE TIEMPO Y FÍSICA ---
FPS = 30
DURATION_SEC = 30
TOTAL_FRAMES = FPS * DURATION_SEC  # 900 Frames
DT = 0.005        # Paso de tiempo de física
STEPS_PER_FRAME = 3 # Cuantos pasos de física calculamos por cada cuadro de video (para acelerar el movimiento)

# --- CONFIGURACIÓN DEL CÚMULO ---
N_BODIES = 10000 
G = 1.0           
CENTER_MASS = 100.0 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🌌 Renderizando Animación: {TOTAL_FRAMES} cuadros ({DURATION_SEC}s) en {device}")

# --- INICIALIZACIÓN (Disco de Acreción) ---
r = torch.rand(N_BODIES, device=device) * 6.0 + 2.0
theta = torch.rand(N_BODIES, device=device) * 2 * 3.14159
positions = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=1)

# Velocidad Orbital
v_mag = (G * CENTER_MASS / r).sqrt()
velocities = torch.stack((-v_mag * torch.sin(theta), v_mag * torch.cos(theta)), dim=1)
velocities += torch.randn(N_BODIES, 2, device=device) * 0.1 # Caos

# --- PREPARAR LA CÁMARA ---
fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
ax.set_facecolor('black')
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.axis('off')

# Objetos gráficos iniciales
scatter_dots = ax.scatter([], [], s=0.2, c='cyan', alpha=0.5)
star = ax.scatter([0], [0], s=50, c='white', marker='*')
title_text = ax.text(0, 11, "Inicializando...", color='white', ha='center')

# --- FUNCIÓN DE ACTUALIZACIÓN (Lo que pasa en cada frame) ---
def update(frame):
    global positions, velocities
    
    # Avanzamos la física varios pasos para que el video no sea en cámara lenta
    for _ in range(STEPS_PER_FRAME):
        dist_sq = positions.pow(2).sum(1, keepdim=True) 
        dist = dist_sq.sqrt()
        force_mag = (G * CENTER_MASS) / (dist_sq + 0.5) 
        acceleration = -positions * (force_mag / dist)
        velocities += acceleration * DT
        positions += velocities * DT
    
    # Actualizar gráfico
    positions_cpu = positions.cpu().numpy()
    scatter_dots.set_offsets(positions_cpu)
    
    # Telemetría en el video
    progress = (frame / TOTAL_FRAMES) * 100
    title_text.set_text(f"Orión-Alaska | Frame: {frame}/{TOTAL_FRAMES} | Progreso: {progress:.1f}%")
    
    # Log en terminal cada 50 frames
    if frame % 50 == 0:
        print(f"🎥 Renderizando frame {frame}/{TOTAL_FRAMES}...")
        
    return scatter_dots, title_text

# --- RENDERIZADO FINAL ---
print("🚀 Iniciando compilación de video MP4...")
start_time = time.time()

ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=True)

# Configuración del encoder (usamos h264 para máxima compatibilidad)
writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Rubin1'), bitrate=5000)
ani.save("orion_simulacion_30s.mp4", writer=writer)

end_time = time.time()
print(f"✅ ¡Video completado! Guardado como 'orion_simulacion_30s.mp4'")
print(f"⏱️ Tiempo de renderizado: {end_time - start_time:.2f} segundos")