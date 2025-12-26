import torch

print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo actual: {torch.cuda.get_device_name(0)}")
    print(f"Cantidad de VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Prueba de estrés pequeña (Multiplicación de Matrices)
    x = torch.rand(5000, 5000).cuda()
    y = torch.rand(5000, 5000).cuda()
    print("¡Realizando cálculo masivo en la RTX 3060...")
    z = torch.matmul(x, y)
    print("¡Cálculo completado! El sistema está listo para la ciencia.")
else:
    print("¡ALERTA! Estamos usando CPU. Revisa los drivers.")