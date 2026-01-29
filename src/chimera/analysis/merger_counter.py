"""
Proyecto Orión - Detector de Fusiones (Merger Counter)
Analiza las trayectorias para detectar cuándo las galaxias colapsan.
Usa KDTree para optimización espacial.
Autor: Chris (Rubin1)
"""

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# Archivos
TRAJ_FILE = "data/processed/trajectory_taichi.npy"
META_FILE = "data/processed/simulation_input.npy"

# Radio crítico de fusión (Si pasan a menos de X parsecs, contamos fusión)
# En el universo real, esto sería el Radio Virial (~10-20 kpc)
MERGER_RADIUS_PC = 15000.0 

def analyze_mergers():
    print("--- 🕵️‍♂️ INICIANDO ANÁLISIS FORENSE DE LA SIMULACIÓN ---")
    
    # Cargar datos
    try:
        traj = np.load(TRAJ_FILE) # (Steps, N, 3)
        meta = np.load(META_FILE, allow_pickle=True).item()
        masses = meta['masses']
    except FileNotFoundError:
        print("❌ Faltan archivos. Corre la simulación GPU primero.")
        return

    n_steps = traj.shape[0]
    n_galaxies = traj.shape[1]
    
    print(f"📊 Analizando {n_galaxies} galaxias a lo largo de {n_steps} pasos de tiempo.")
    print(f"   Criterio de fusión: Distancia < {MERGER_RADIUS_PC/1000:.1f} kpc")

    # Vamos a analizar solo el ÚLTIMO cuadro para ver cómo terminó todo
    # (Hacerlo paso a paso es posible pero tardado, empecemos por el final)
    final_pos = traj[-1] # (N, 3) en Parsecs
    
    # Construir un árbol espacial (KDTree) para búsquedas rápidas
    tree = cKDTree(final_pos)
    
    # Buscar grupos: "Dame todos los vecinos a menos de X distancia"
    # query_ball_tree encuentra clusters automáticamente
    merger_groups = tree.query_ball_tree(tree, r=MERGER_RADIUS_PC)
    
    # merger_groups es una lista de listas. Ej: [[0, 1], [1, 0], [2], [3, 4, 5]...]
    # Necesitamos limpiar duplicados y encontrar los grupos únicos.
    
    visited = set()
    clusters = []
    
    for i, neighbors in enumerate(merger_groups):
        if i not in visited:
            # Encontramos un nuevo grupo (o galaxia solitaria)
            # Usamos un algoritmo de "Inundación" (BFS) para encontrar todo el cluster conectado
            current_cluster = set()
            stack = [i]
            
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    current_cluster.add(node)
                    # Añadir vecinos de este nodo a la pila
                    stack.extend(merger_groups[node])
            
            clusters.append(list(current_cluster))

    # --- RESULTADOS ---
    n_mergers = 0
    max_mass = 0
    monster_cluster = []

    print("\n--- RESULTADOS DEL COLAPSO ---")
    
    for cluster in clusters:
        cluster_size = len(cluster)
        
        if cluster_size > 1:
            n_mergers += 1
            
            # Calcular masa total del monstruo resultante
            cluster_mass = np.sum(masses[cluster])
            
            if cluster_mass > max_mass:
                max_mass = cluster_mass
                monster_cluster = cluster
            
            # Solo imprimir fusiones grandes
            if cluster_size > 5:
                print(f"⚠️ FUSIÓN MASIVA DETECTADA: {cluster_size} galaxias colapsaron en un solo objeto.")
                print(f"   Masa combinada: {cluster_mass:.2e} M_sol")

    print("\n" + "="*30)
    print(f"✅ Total de objetos finales: {len(clusters)} (de {n_galaxies} iniciales)")
    print(f"🔥 Eventos de fusión detectados: {n_mergers}")
    print(f"👑 EL MONSTRUO (Agujero Negro Semilla más grande):")
    print(f"   Compuesto por: {len(monster_cluster)} galaxias")
    print(f"   Masa Final: {max_mass:.4e} Masas Solares")
    print("="*30)
    
    # Validación de Hipótesis
    if max_mass > 1e12: # Umbral arbitrario para "Semilla Supermasiva"
        print("\n🚀 CONCLUSIÓN: ¡La densidad fue suficiente! Hipótesis viable.")
    else:
        print("\n📉 CONCLUSIÓN: Crecimiento insuficiente. Necesitamos más densidad o más tiempo.")

if __name__ == "__main__":
    analyze_mergers()