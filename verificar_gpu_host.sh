#!/bin/bash
# Script para verificar GPU NVIDIA en el host

echo "=========================================="
echo "VERIFICACIÓN DE GPU NVIDIA"
echo "=========================================="

echo ""
echo "1. Verificando GPU con lspci:"
lspci | grep -i nvidia

echo ""
echo "2. Verificando drivers NVIDIA:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠ nvidia-smi no encontrado. Necesitas instalar drivers NVIDIA."
fi

echo ""
echo "3. Verificando TensorFlow con GPU:"
source ~/tensorflow/bin/activate 2>/dev/null || echo "⚠ Entorno virtual no encontrado"
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'CUDA disponible: {tf.test.is_built_with_cuda()}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs detectadas: {len(gpus)}')
if len(gpus) > 0:
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu}')
        print(f'  Nombre: {tf.config.experimental.get_device_details(gpu)}')
else:
    print('⚠ No se detectaron GPUs')
" 2>&1

echo ""
echo "=========================================="



