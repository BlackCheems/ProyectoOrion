import torch
import time
import sys

def print_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"   [VRAM] Asignada: {allocated:.2f} GB | Reservada: {reserved:.2f} GB")

def stress_test():
    if not torch.cuda.is_available():
        print("❌ CRÍTICO: CUDA no detectado. Abortando.")
        return

    device = torch.device("cuda")
    print(f"🚀 Iniciando Secuencia de Estrés en: {torch.cuda.get_device_name(0)}")
    print_memory()

    try:
        # FASE 1: Llenado de Memoria (Allocation)
        print("\n🔹 FASE 1: Inyectando Tensores Masivos...")
        # Crear dos matrices de 10000x10000 floats (aprox 400MB cada una) y subirlas a VRAM
        # Vamos a crear una lista para acumular basura en VRAM hasta llenarla
        tensors = []
        target_gb = 10.0 # Intentar llenar 10GB seguros
        current_gb = 0
        
        while current_gb < target_gb:
            # Matriz de 5000x5000 * 4 bytes = ~100MB
            size = 5000 
            t = torch.randn(size, size, device=device)
            tensors.append(t)
            current_gb += (size*size*4) / (1024**3)
            sys.stdout.write(f"\r   -> Inyectando... {current_gb:.2f} GB cargados")
            sys.stdout.flush()
            time.sleep(0.1) # Pequeña pausa para ver subir la gráfica
            
        print("\n✅ Carga de Memoria Completa.")
        print_memory()

        # FASE 2: Combustión (Cálculo Matemático)
        print("\n🔥 FASE 2: Ignición de Tensor Cores (Multiplicación de Matrices)...")
        print("   Presiona CTRL+C para detener el reactor.")
        
        start_time = time.time()
        iteracion = 0
        while True:
            # Tomamos dos tensores de la lista y los multiplicamos
            # Esto fuerza a la GPU a calcular, no solo almacenar
            res = torch.matmul(tensors[0], tensors[-1]) 
            torch.cuda.synchronize() # Esperar a que termine el cálculo
            
            if iteracion % 10 == 0:
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r   -> Ciclo {iteracion} completado | Tiempo: {elapsed:.1f}s | GPU a plena carga")
                sys.stdout.flush()
            iteracion += 1

    except KeyboardInterrupt:
        print("\n\n🛑 ABORTADO POR EL USUARIO. Enfriando reactores...")
    except RuntimeError as e:
        print(f"\n\n💥 ERROR DE MEMORIA: {e}")
    finally:
        # Limpieza
        del tensors
        torch.cuda.empty_cache()
        print("📉 VRAM liberada. Prueba finalizada.")

if __name__ == "__main__":
    stress_test()
