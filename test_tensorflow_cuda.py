"""
Script de prueba para verificar TensorFlow y CUDA
"""

import logging
import os

# Deshabilitar mensajes de advertencia de TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np

print("="*60)
print("PRUEBA DE TENSORFLOW Y CUDA")
print("="*60)

# Información de TensorFlow
print(f"\n✓ TensorFlow versión: {tf.__version__}")
print(f"✓ TensorFlow compilado con CUDA: {tf.test.is_built_with_cuda()}")

# Verificar GPUs disponibles
gpus = tf.config.list_physical_devices('GPU')
print(f"\n✓ GPUs físicas detectadas: {len(gpus)}")

if len(gpus) > 0:
    print("\nDetalles de las GPUs:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        # Obtener más detalles de la GPU
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"    Detalles: {gpu_details}")
        except:
            pass
else:
    print("\n⚠ No se detectaron GPUs CUDA")
    print("  Esto puede deberse a:")
    print("  - No hay tarjeta gráfica NVIDIA instalada")
    print("  - Los drivers de NVIDIA no están instalados")
    print("  - CUDA no está correctamente configurado")

# Verificar si TensorFlow puede usar GPU
print(f"\n✓ TensorFlow puede usar GPU: {tf.test.is_gpu_available()}")

# Prueba simple de operación
print("\n" + "="*60)
print("PRUEBA DE OPERACIÓN")
print("="*60)

# Crear un tensor simple
print("\nCreando tensores de prueba...")
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Realizar una operación
print("Realizando multiplicación de matrices...")
c = tf.matmul(a, b)

print(f"\nResultado:")
print(c.numpy())

# Prueba con GPU si está disponible
if len(gpus) > 0:
    print("\n" + "="*60)
    print("PRUEBA CON GPU")
    print("="*60)
    
    try:
        with tf.device('/GPU:0'):
            print("\nEjecutando operación en GPU...")
            a_gpu = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b_gpu = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c_gpu = tf.matmul(a_gpu, b_gpu)
            print(f"Resultado (GPU):")
            print(c_gpu.numpy())
            print("✓ Operación en GPU completada exitosamente")
    except Exception as e:
        print(f"✗ Error al usar GPU: {e}")
else:
    print("\n⚠ No se puede probar GPU (no hay GPUs disponibles)")

print("\n" + "="*60)
print("PRUEBA COMPLETADA")
print("="*60)


