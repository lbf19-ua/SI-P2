import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Deshabilitar mensajes de advertencia de TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


def cargar_y_preprocesar_cifar10():
    # Cargar datos
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    
    # Normalizar valores de píxeles al rango [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Aplanar imágenes para MLP (32x32x3 = 3072 características)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Convertir etiquetas a formato categórico one-hot
    Y_train = to_categorical(Y_train, 10)
    Y_test = to_categorical(Y_test, 10)
    
    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de Y_train: {Y_train.shape}")
    print(f"Forma de X_test: {X_test.shape}")
    print(f"Forma de Y_test: {Y_test.shape}")
    
    return X_train, Y_train, X_test, Y_test


def plot_evolucion_entrenamiento(history, titulo="Evolución del entrenamiento"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfica de pérdida
    ax1.plot(history.history['loss'], label='Train Loss', marker='o')
    ax1.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Evolución de la Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica de accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Tasa de Acierto')
    ax2.set_title('Evolución de la Tasa de Acierto')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()


def plot_comparacion_modelos(resultados, titulo="Comparación de modelos"):
    nombres = [r['nombre'] for r in resultados]
    tiempos = [r['tiempo'] for r in resultados]
    accuracies = [r['accuracy'] for r in resultados]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfica de tiempo
    ax1.bar(nombres, tiempos, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Modelo')
    ax1.set_ylabel('Tiempo de Entrenamiento (segundos)')
    ax1.set_title('Tiempo de Entrenamiento')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfica de accuracy
    ax2.bar(nombres, accuracies, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.set_xlabel('Modelo')
    ax2.set_ylabel('Tasa de Acierto (Test)')
    ax2.set_title('Tasa de Acierto Final')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()


def plot_matriz_confusion(y_true, y_pred, titulo="Matriz de Confusión"):
    # Convertir de one-hot a índices si es necesario
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Nombres de las clases de CIFAR-10
    clases = ['Avión', 'Coche', 'Pájaro', 'Gato', 'Ciervo', 
              'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']
    
    # Visualizar
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clases, yticklabels=clases)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def probar_MLP1(X_train, X_test, Y_train, Y_test):
    print("\n" + "="*60)
    print("TAREA MLP1: Definir, entrenar y evaluar un MLP con Keras")
    print("="*60)
    
    model = Sequential([
        Dense(48, activation='sigmoid', input_shape=(3072,), name='capa_oculta'),  # MLP1: 48 neuronas, sigmoid
        Dense(10, activation='softmax', name='capa_salida')  # MLP1: 10 neuronas, softmax
    ])
    
    model.compile(
        optimizer='adam',  # MLP1: Optimizador Adam (requisito del enunciado)
        loss='categorical_crossentropy',  # MLP1: Loss categorical_crossentropy (requisito del enunciado)
        metrics=['accuracy']  # MLP1: Métrica accuracy (requisito del enunciado)
    )
    
    print("\nResumen del modelo:")
    model.summary()
    
    print("\nIniciando entrenamiento...")
    inicio_tiempo = time.time()
    
    history = model.fit(
        X_train, Y_train,
        validation_split=0.1,  # MLP1: validation_split=0.1 (requisito del enunciado)
        batch_size=32,  # MLP1: batch_size=32 (requisito del enunciado)
        epochs=10,  # MLP1: epochs=10 (requisito del enunciado)
        verbose=1
    )
    
    tiempo_entrenamiento = time.time() - inicio_tiempo
    print(f"\nTiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")
    
    print("\nEvaluando modelo con conjunto de test...")
    resultados_eval = model.evaluate(X_test, Y_test, verbose=1)
    test_loss = resultados_eval[0]
    test_accuracy = resultados_eval[1]
    
    print(f"\nResultados en conjunto de test:")
    print(f"  Pérdida: {test_loss:.4f}")
    print(f"  Tasa de acierto: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    y_pred = model.predict(X_test, verbose=0)
    
    # Mostrar gráficas de evolución del entrenamiento (requisito del enunciado)
    plot_evolucion_entrenamiento(history, "MLP1: Evolución del Entrenamiento")
    
    # Mostrar matriz de confusión (requisito del enunciado)
    plot_matriz_confusion(Y_test, y_pred, "MLP1: Matriz de Confusión")
    
    # Preparar resultados del MLP1
    resultados = {
        'nombre': 'MLP1',
        'tiempo': tiempo_entrenamiento,
        'accuracy': test_accuracy,
        'loss': test_loss
    }
    
    return model, history, resultados


def promediar_historias(historias):
    # MLP2: Inicializar diccionario para almacenar promedios
    promedios = {}
    
    # MLP2: Verificar que hay historias para promediar
    if not historias:
        raise ValueError("La lista de historias está vacía. No se pueden promediar resultados.")
    
    # MLP2: Obtener todas las claves de las métricas
    metricas = list(historias[0].history.keys())
    
    # MLP2: Promediar cada métrica
    for metrica in metricas:
        # Obtener todos los valores de esta métrica de todas las repeticiones
        valores = [h.history[metrica] for h in historias]
        
        # Asegurar que todas tienen la misma longitud (rellenar con el último valor si es necesario)
        max_len = max(len(v) for v in valores)
        valores_padded = []
        for v in valores:
            if len(v) < max_len:
                v = v + [v[-1]] * (max_len - len(v))
            valores_padded.append(v)
        
        # Promediar
        promedios[metrica] = np.mean(valores_padded, axis=0).tolist()
    
    return promedios


def plot_evolucion_mlp2(historias_promediadas, titulo="MLP2: Evolución del Entrenamiento (Promediado)"):
    # MLP2: Crear figura con dos subplots y ejes Y secundarios
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # MLP2: Eje Y izquierdo para Loss
    color_loss = 'tab:red'
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Pérdida', color=color_loss, fontsize=12)
    ax1.plot(historias_promediadas['loss'], label='Train Loss', 
             color=color_loss, marker='o', linestyle='-', linewidth=2)
    ax1.plot(historias_promediadas['val_loss'], label='Validation Loss', 
             color='tab:orange', marker='s', linestyle='--', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # MLP2: Eje Y derecho para Accuracy
    ax2 = ax1.twinx()
    color_acc = 'tab:blue'
    ax2.set_ylabel('Tasa de Acierto', color=color_acc, fontsize=12)
    ax2.plot(historias_promediadas['accuracy'], label='Train Accuracy', 
             color=color_acc, marker='o', linestyle='-', linewidth=2)
    ax2.plot(historias_promediadas['val_accuracy'], label='Validation Accuracy', 
             color='tab:cyan', marker='s', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.legend(loc='upper right')
    
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def probar_MLP2(X_train, X_test, Y_train, Y_test, num_repeticiones=5, epochs_max=30):
    print("\n" + "="*60)
    print("TAREA MLP2: Ajustar el valor del parámetro epochs")
    print("="*60)
    print("\n[MLP2] Modelo: Misma arquitectura que MLP1 (48 neuronas sigmoid, 10 softmax)")
    print(f"\n[MLP2] Realizando {num_repeticiones} repeticiones del entrenamiento...")
    print(f"[MLP2] Épocas máximas: {epochs_max}")
    
    historias = []
    modelos = []
    tiempos = []
    
    for i in range(num_repeticiones):
        print(f"\n[MLP2] Repetición {i+1}/{num_repeticiones}...")
        
        # MLP2: Crear un nuevo modelo para cada repetición (inicialización aleatoria diferente)
        model = Sequential([
            Dense(48, activation='sigmoid', input_shape=(3072,), name='capa_oculta'),
            Dense(10, activation='softmax', name='capa_salida')
        ])
        
        # MLP2: Compilar el modelo (misma configuración que MLP1)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # MLP2: Entrenar el modelo (sin EarlyStopping inicialmente para ver la evolución completa)
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            validation_split=0.1,
            batch_size=32,
            epochs=epochs_max,
            verbose=0  # Silencioso para múltiples repeticiones
        )
        tiempo = time.time() - inicio
        
        historias.append(history)
        modelos.append(model)
        tiempos.append(tiempo)
        
        print(f"[MLP2] Repetición {i+1} completada - Tiempo: {tiempo:.2f}s")
    
    print("\n[MLP2] Promediando resultados de las repeticiones...")
    historias_promediadas = promediar_historias(historias)
    
    print("\n[MLP2] Generando gráficas de evolución del entrenamiento...")
    plot_evolucion_mlp2(historias_promediadas, 
                        f"MLP2: Evolución del Entrenamiento ({num_repeticiones} repeticiones promediadas)")
    
    val_acc = historias_promediadas['val_accuracy']
    val_loss = historias_promediadas['val_loss']
    
    # MLP2: Encontrar la época con mejor val_accuracy
    mejor_epoca_acc = np.argmax(val_acc)
    mejor_val_acc = val_acc[mejor_epoca_acc]
    
    # MLP2: Encontrar la época con mejor val_loss
    mejor_epoca_loss = np.argmin(val_loss)
    mejor_val_loss = val_loss[mejor_epoca_loss]
    
    print(f"\n[MLP2] Análisis de resultados promediados:")
    print(f"  - Mejor val_accuracy: {mejor_val_acc:.4f} en época {mejor_epoca_acc + 1}")
    print(f"  - Mejor val_loss: {mejor_val_loss:.4f} en época {mejor_epoca_loss + 1}")
    print(f"  - Tiempo promedio por repetición: {np.mean(tiempos):.2f}s")
    
    # MLP2: Detectar sobreentrenamiento (val_loss aumenta después del mínimo)
    if mejor_epoca_loss < len(val_loss) - 1:
        sobreentrenamiento = val_loss[-1] > mejor_val_loss * 1.1  # 10% peor
        if sobreentrenamiento:
            print(f"  - Sobreentrenamiento detectado: val_loss empeora después de época {mejor_epoca_loss + 1}")
        else:
            print(f"  - ✓ No se detectó sobreentrenamiento significativo")
    
    print("\n[MLP2] Probando EarlyStopping con diferentes configuraciones...")
    
    configuraciones_early_stopping = [
        {'monitor': 'val_accuracy', 'patience': 3, 'mode': 'max', 'restore_best_weights': True},
        {'monitor': 'val_loss', 'patience': 5, 'mode': 'min', 'restore_best_weights': True},
        {'monitor': 'val_accuracy', 'patience': 5, 'mode': 'max', 'restore_best_weights': True},
    ]
    
    resultados_early_stopping = []
    
    for idx, config in enumerate(configuraciones_early_stopping):
        print(f"\n[MLP2] EarlyStopping Config {idx+1}: monitor={config['monitor']}, patience={config['patience']}")
        
        # MLP2: Crear modelo para esta configuración
        model_es = Sequential([
            Dense(48, activation='sigmoid', input_shape=(3072,), name='capa_oculta'),
            Dense(10, activation='softmax', name='capa_salida')
        ])
        model_es.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # MLP2: Crear callback EarlyStopping
        early_stop = EarlyStopping(
            monitor=config['monitor'],
            patience=config['patience'],
            mode=config['mode'],
            restore_best_weights=config['restore_best_weights'],
            verbose=1
        )
        
        # MLP2: Entrenar con EarlyStopping
        inicio = time.time()
        history_es = model_es.fit(
            X_train, Y_train,
            validation_split=0.1,
            batch_size=32,
            epochs=epochs_max,
            callbacks=[early_stop],
            verbose=0
        )
        tiempo_es = time.time() - inicio
        
        # MLP2: Evaluar con test
        resultados_eval = model_es.evaluate(X_test, Y_test, verbose=0)
        test_acc = resultados_eval[1]
        
        epocas_entrenadas = len(history_es.history['loss'])
        
        resultados_early_stopping.append({
            'config': config,
            'epocas': epocas_entrenadas,
            'test_accuracy': test_acc,
            'tiempo': tiempo_es,
            'history': history_es
        })
        
        print(f"  [MLP2] Entrenamiento detenido en época {epocas_entrenadas}")
        print(f"  [MLP2] Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  [MLP2] Tiempo: {tiempo_es:.2f}s")
    
    # MLP2: Mostrar comparación de configuraciones de EarlyStopping
    print("\n[MLP2] Resumen de configuraciones EarlyStopping:")
    for idx, res in enumerate(resultados_early_stopping):
        print(f"  Config {idx+1}: {res['epocas']} épocas, test_acc={res['test_accuracy']:.4f}, tiempo={res['tiempo']:.2f}s")
    
    # MLP2: Preparar resultados finales
    resultados = {
        'historias_promediadas': historias_promediadas,
        'mejor_epoca_acc': mejor_epoca_acc + 1,
        'mejor_val_acc': mejor_val_acc,
        'mejor_epoca_loss': mejor_epoca_loss + 1,
        'mejor_val_loss': mejor_val_loss,
        'resultados_early_stopping': resultados_early_stopping,
        'tiempo_promedio': np.mean(tiempos)
    }
    
    return resultados


def probar_MLP3(X_train, X_test, Y_train, Y_test, batch_sizes=None, epochs_max=50, patience=5):
    print("\n" + "="*60)
    print("TAREA MLP3: Ajustar el valor del parámetro batch_size")
    print("="*60)
    
    if batch_sizes is None:
        batch_sizes = [16, 32, 64, 128, 256]  # MLP3: Valores típicos de batch_size
    
    print(f"\n[MLP3] Valores de batch_size a probar: {batch_sizes}")
    print(f"[MLP3] Épocas máximas: {epochs_max}")
    print(f"[MLP3] EarlyStopping patience: {patience}")
    
    resultados_batch_sizes = []
    
    for batch_size in batch_sizes:
        print(f"\n[MLP3] Probando batch_size = {batch_size}...")
        
        # MLP3: Crear modelo con misma arquitectura que MLP1
        model = Sequential([
            Dense(48, activation='sigmoid', input_shape=(3072,), name='capa_oculta'),
            Dense(10, activation='softmax', name='capa_salida')
        ])
        
        # MLP3: Compilar modelo (misma configuración que MLP1)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # MLP3: Crear callback EarlyStopping (aprendido en MLP2)
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=0
        )
        
        # MLP3: Entrenar modelo con este batch_size
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            validation_split=0.1,
            batch_size=batch_size,  # MLP3: Variable a probar
            epochs=epochs_max,
            callbacks=[early_stop],
            verbose=0  # Silencioso para múltiples pruebas
        )
        tiempo_entrenamiento = time.time() - inicio
        
        # MLP3: Evaluar con test
        resultados_eval = model.evaluate(X_test, Y_test, verbose=0)
        test_loss = resultados_eval[0]
        test_accuracy = resultados_eval[1]
        
        epocas_entrenadas = len(history.history['loss'])
        
        # MLP3: Obtener mejor val_accuracy durante el entrenamiento
        mejor_val_acc = max(history.history['val_accuracy'])
        mejor_epoca = np.argmax(history.history['val_accuracy']) + 1
        
        # MLP3: Calcular actualizaciones por época
        num_muestras_entrenamiento = int(X_train.shape[0] * 0.9)  # 90% (validation_split=0.1)
        actualizaciones_por_epoca = num_muestras_entrenamiento // batch_size
        
        resultado = {
            'batch_size': batch_size,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'tiempo': tiempo_entrenamiento,
            'epocas': epocas_entrenadas,
            'mejor_val_acc': mejor_val_acc,
            'mejor_epoca': mejor_epoca,
            'actualizaciones_por_epoca': actualizaciones_por_epoca,
            'history': history,
            'modelo': model
        }
        
        resultados_batch_sizes.append(resultado)
        
        print(f"  [MLP3] Batch size {batch_size}: {epocas_entrenadas} épocas, "
              f"test_acc={test_accuracy:.4f}, tiempo={tiempo_entrenamiento:.2f}s, "
              f"updates/época={actualizaciones_por_epoca}")
    
    print("\n[MLP3] Análisis de resultados:")
    print("-" * 60)
    
    # MLP3: Crear tabla de resultados
    print(f"{'Batch Size':<12} {'Test Acc':<12} {'Tiempo (s)':<12} {'Épocas':<10} {'Updates/Época':<15}")
    print("-" * 60)
    for res in resultados_batch_sizes:
        print(f"{res['batch_size']:<12} {res['test_accuracy']:<12.4f} "
              f"{res['tiempo']:<12.2f} {res['epocas']:<10} {res['actualizaciones_por_epoca']:<15}")
    
    # MLP3: Encontrar mejor batch_size según diferentes criterios
    mejor_por_accuracy = max(resultados_batch_sizes, key=lambda x: x['test_accuracy'])
    mejor_por_tiempo = min(resultados_batch_sizes, key=lambda x: x['tiempo'])
    
    # MLP3: Calcular eficiencia (accuracy/tiempo)
    for res in resultados_batch_sizes:
        res['eficiencia'] = res['test_accuracy'] / res['tiempo']
    
    mejor_por_eficiencia = max(resultados_batch_sizes, key=lambda x: x['eficiencia'])
    
    print("\n[MLP3] Resumen de mejores resultados:")
    print(f"  - Mejor accuracy: batch_size={mejor_por_accuracy['batch_size']} "
          f"(accuracy={mejor_por_accuracy['test_accuracy']:.4f})")
    print(f"  - Más rápido: batch_size={mejor_por_tiempo['batch_size']} "
          f"(tiempo={mejor_por_tiempo['tiempo']:.2f}s)")
    print(f"  - Mejor eficiencia (acc/tiempo): batch_size={mejor_por_eficiencia['batch_size']} "
          f"(eficiencia={mejor_por_eficiencia['eficiencia']:.6f})")
    
    print("\n[MLP3] Generando gráficas comparativas...")
    
    # MLP3: Gráfica comparativa de accuracy y tiempo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    batch_vals = [r['batch_size'] for r in resultados_batch_sizes]
    accuracies = [r['test_accuracy'] for r in resultados_batch_sizes]
    tiempos = [r['tiempo'] for r in resultados_batch_sizes]
    
    # Gráfica de accuracy
    ax1.plot(batch_vals, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs Batch Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)  # Escala logarítmica para batch_size
    
    # Gráfica de tiempo
    ax2.plot(batch_vals, tiempos, marker='s', linestyle='--', linewidth=2, 
             markersize=8, color='tab:red')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Tiempo de Entrenamiento (segundos)', fontsize=12)
    ax2.set_title('Tiempo de Entrenamiento vs Batch Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.suptitle('MLP3: Comparación de Batch Sizes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # MLP3: Gráfica de curvas de entrenamiento para diferentes batch_sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de accuracy por época
    for res in resultados_batch_sizes:
        ax1.plot(res['history'].history['val_accuracy'], 
                label=f"Batch={res['batch_size']}", linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Evolución de Validation Accuracy por Batch Size', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica de loss por época
    for res in resultados_batch_sizes:
        ax2.plot(res['history'].history['val_loss'], 
                label=f"Batch={res['batch_size']}", linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Evolución de Validation Loss por Batch Size', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('MLP3: Evolución del Entrenamiento por Batch Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # MLP3: Preparar resultados finales
    resultados = {
        'resultados_batch_sizes': resultados_batch_sizes,
        'mejor_por_accuracy': mejor_por_accuracy['batch_size'],
        'mejor_accuracy': mejor_por_accuracy['test_accuracy'],
        'mejor_por_tiempo': mejor_por_tiempo['batch_size'],
        'mejor_tiempo': mejor_por_tiempo['tiempo'],
        'mejor_por_eficiencia': mejor_por_eficiencia['batch_size'],
        'mejor_eficiencia': mejor_por_eficiencia['eficiencia']
    }
    
    return resultados


def probar_MLP4(X_train, X_test, Y_train, Y_test, activaciones=None, epochs_max=30, patience=5):
    print("\n" + "="*60)
    print("TAREA MLP4: Probar diferentes funciones de activación")
    print("="*60)
    
    if activaciones is None:
        activaciones = ['sigmoid', 'relu', 'tanh']  # MLP4: Funciones de activación comunes
    
    print(f"\n[MLP4] Funciones de activación a probar: {activaciones}")
    print(f"[MLP4] Épocas máximas: {epochs_max}")
    print(f"[MLP4] EarlyStopping patience: {patience}")
    
    resultados_activaciones = []
    
    for activacion in activaciones:
        print(f"\n[MLP4] Probando activación = {activacion}...")
        
        # MLP4: Crear modelo con misma arquitectura que MLP1 pero diferente activación
        model = Sequential([
            Dense(48, activation=activacion, input_shape=(3072,), name='capa_oculta'),
            Dense(10, activation='softmax', name='capa_salida')
        ])
        
        # MLP4: Compilar modelo (misma configuración que MLP1)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # MLP4: Crear callback EarlyStopping
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=0
        )
        
        # MLP4: Entrenar modelo con esta activación
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            validation_split=0.1,
            batch_size=32,
            epochs=epochs_max,
            callbacks=[early_stop],
            verbose=0
        )
        tiempo_entrenamiento = time.time() - inicio
        
        # MLP4: Evaluar con test
        resultados_eval = model.evaluate(X_test, Y_test, verbose=0)
        test_loss = resultados_eval[0]
        test_accuracy = resultados_eval[1]
        
        epocas_entrenadas = len(history.history['loss'])
        mejor_val_acc = max(history.history['val_accuracy'])
        
        resultado = {
            'activacion': activacion,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'tiempo': tiempo_entrenamiento,
            'epocas': epocas_entrenadas,
            'mejor_val_acc': mejor_val_acc,
            'history': history,
            'modelo': model
        }
        
        resultados_activaciones.append(resultado)
        
        print(f"  [MLP4] {activacion}: {epocas_entrenadas} épocas, "
              f"test_acc={test_accuracy:.4f}, tiempo={tiempo_entrenamiento:.2f}s")
    
    print("\n[MLP4] Análisis de resultados:")
    print("-" * 60)
    
    print(f"{'Activación':<15} {'Test Acc':<12} {'Tiempo (s)':<12} {'Épocas':<10}")
    print("-" * 60)
    for res in resultados_activaciones:
        print(f"{res['activacion']:<15} {res['test_accuracy']:<12.4f} "
              f"{res['tiempo']:<12.2f} {res['epocas']:<10}")
    
    # MLP4: Encontrar mejor activación
    mejor_por_accuracy = max(resultados_activaciones, key=lambda x: x['test_accuracy'])
    mejor_por_tiempo = min(resultados_activaciones, key=lambda x: x['tiempo'])
    
    print("\n[MLP4] Resumen de mejores resultados:")
    print(f"  - Mejor accuracy: activación={mejor_por_accuracy['activacion']} "
          f"(accuracy={mejor_por_accuracy['test_accuracy']:.4f})")
    print(f"  - Más rápido: activación={mejor_por_tiempo['activacion']} "
          f"(tiempo={mejor_por_tiempo['tiempo']:.2f}s)")

    print("\n[MLP4] Generando gráficas comparativas...")
    
    # MLP4: Gráfica comparativa de accuracy y tiempo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    activaciones_nombres = [r['activacion'] for r in resultados_activaciones]
    accuracies = [r['test_accuracy'] for r in resultados_activaciones]
    tiempos = [r['tiempo'] for r in resultados_activaciones]
    
    # Gráfica de accuracy
    ax1.bar(activaciones_nombres, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'], 
            edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Función de Activación', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy por Función de Activación', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfica de tiempo
    ax2.bar(activaciones_nombres, tiempos, color=['skyblue', 'lightgreen', 'lightcoral'],
            edgecolor='navy', alpha=0.7)
    ax2.set_xlabel('Función de Activación', fontsize=12)
    ax2.set_ylabel('Tiempo de Entrenamiento (segundos)', fontsize=12)
    ax2.set_title('Tiempo de Entrenamiento por Función de Activación', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MLP4: Comparación de Funciones de Activación', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # MLP4: Gráfica de curvas de entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de accuracy por época
    for res in resultados_activaciones:
        ax1.plot(res['history'].history['val_accuracy'], 
                label=f"{res['activacion']}", linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Evolución de Validation Accuracy por Activación', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica de loss por época
    for res in resultados_activaciones:
        ax2.plot(res['history'].history['val_loss'], 
                label=f"{res['activacion']}", linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Evolución de Validation Loss por Activación', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('MLP4: Evolución del Entrenamiento por Activación', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # MLP4: Preparar resultados finales
    resultados = {
        'resultados_activaciones': resultados_activaciones,
        'mejor_activacion': mejor_por_accuracy['activacion'],
        'mejor_accuracy': mejor_por_accuracy['test_accuracy']
    }
    
    return resultados


def probar_MLP5(X_train, X_test, Y_train, Y_test, num_neuronas=None, epochs_max=30, patience=5):
    print("\n" + "="*60)
    print("TAREA MLP5: Ajustar el número de neuronas")
    print("="*60)
    
    if num_neuronas is None:
        num_neuronas = [24, 48, 96, 192, 384]  # MLP5: Valores típicos de neuronas
    
    print(f"\n[MLP5] Números de neuronas a probar: {num_neuronas}")
    print(f"[MLP5] Épocas máximas: {epochs_max}")
    print(f"[MLP5] EarlyStopping patience: {patience}")
    
    resultados_neuronas = []
    
    for neuronas in num_neuronas:
        print(f"\n[MLP5] Probando {neuronas} neuronas...")
        
        # MLP5: Crear modelo con diferente número de neuronas
        model = Sequential([
            Dense(neuronas, activation='sigmoid', input_shape=(3072,), name='capa_oculta'),
            Dense(10, activation='softmax', name='capa_salida')
        ])
        
        # MLP5: Compilar modelo
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # MLP5: Crear callback EarlyStopping
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=0
        )
        
        # MLP5: Entrenar modelo
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            validation_split=0.1,
            batch_size=32,
            epochs=epochs_max,
            callbacks=[early_stop],
            verbose=0
        )
        tiempo_entrenamiento = time.time() - inicio
        
        # MLP5: Evaluar con test
        resultados_eval = model.evaluate(X_test, Y_test, verbose=0)
        test_loss = resultados_eval[0]
        test_accuracy = resultados_eval[1]
        
        epocas_entrenadas = len(history.history['loss'])
        mejor_val_acc = max(history.history['val_accuracy'])
        
        # MLP5: Calcular número de parámetros
        num_parametros = model.count_params()
        
        resultado = {
            'neuronas': neuronas,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'tiempo': tiempo_entrenamiento,
            'epocas': epocas_entrenadas,
            'mejor_val_acc': mejor_val_acc,
            'num_parametros': num_parametros,
            'history': history,
            'modelo': model
        }
        
        resultados_neuronas.append(resultado)
        
        print(f"  [MLP5] {neuronas} neuronas: {epocas_entrenadas} épocas, "
              f"test_acc={test_accuracy:.4f}, params={num_parametros}, tiempo={tiempo_entrenamiento:.2f}s")
    
    print("\n[MLP5] Análisis de resultados:")
    print("-" * 70)
    
    print(f"{'Neuronas':<12} {'Test Acc':<12} {'Parámetros':<15} {'Tiempo (s)':<12} {'Épocas':<10}")
    print("-" * 70)
    for res in resultados_neuronas:
        print(f"{res['neuronas']:<12} {res['test_accuracy']:<12.4f} "
              f"{res['num_parametros']:<15} {res['tiempo']:<12.2f} {res['epocas']:<10}")
    
    # MLP5: Encontrar mejor configuración
    mejor_por_accuracy = max(resultados_neuronas, key=lambda x: x['test_accuracy'])
    
    # MLP5: Calcular eficiencia (accuracy/parámetros)
    for res in resultados_neuronas:
        res['eficiencia'] = res['test_accuracy'] / (res['num_parametros'] / 1000)  # accuracy por 1K params
    
    mejor_por_eficiencia = max(resultados_neuronas, key=lambda x: x['eficiencia'])
    
    print("\n[MLP5] Resumen de mejores resultados:")
    print(f"  - Mejor accuracy: {mejor_por_accuracy['neuronas']} neuronas "
          f"(accuracy={mejor_por_accuracy['test_accuracy']:.4f}, params={mejor_por_accuracy['num_parametros']})")
    print(f"  - Mejor eficiencia (acc/1K params): {mejor_por_eficiencia['neuronas']} neuronas "
          f"(eficiencia={mejor_por_eficiencia['eficiencia']:.6f})")
    
    print("\n[MLP5] Generando gráficas comparativas...")
    
    # MLP5: Gráfica comparativa
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    neuronas_vals = [r['neuronas'] for r in resultados_neuronas]
    accuracies = [r['test_accuracy'] for r in resultados_neuronas]
    tiempos = [r['tiempo'] for r in resultados_neuronas]
    parametros = [r['num_parametros'] for r in resultados_neuronas]
    
    # Gráfica de accuracy vs neuronas
    ax1.plot(neuronas_vals, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Neuronas', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs Número de Neuronas', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Gráfica de tiempo vs neuronas
    ax2.plot(neuronas_vals, tiempos, marker='s', linestyle='--', linewidth=2, markersize=8, color='tab:red')
    ax2.set_xlabel('Número de Neuronas', fontsize=12)
    ax2.set_ylabel('Tiempo de Entrenamiento (segundos)', fontsize=12)
    ax2.set_title('Tiempo vs Número de Neuronas', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # Gráfica de parámetros vs neuronas
    ax3.plot(neuronas_vals, parametros, marker='^', linestyle=':', linewidth=2, markersize=8, color='tab:green')
    ax3.set_xlabel('Número de Neuronas', fontsize=12)
    ax3.set_ylabel('Número de Parámetros', fontsize=12)
    ax3.set_title('Parámetros vs Número de Neuronas', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    
    # Gráfica de accuracy vs parámetros
    ax4.scatter(parametros, accuracies, s=100, alpha=0.6, c=range(len(parametros)), cmap='viridis')
    ax4.set_xlabel('Número de Parámetros', fontsize=12)
    ax4.set_ylabel('Test Accuracy', fontsize=12)
    ax4.set_title('Accuracy vs Número de Parámetros', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    for i, n in enumerate(neuronas_vals):
        ax4.annotate(f'{n}N', (parametros[i], accuracies[i]), fontsize=9)
    
    plt.suptitle('MLP5: Comparación de Números de Neuronas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # MLP5: Preparar resultados finales
    resultados = {
        'resultados_neuronas': resultados_neuronas,
        'mejor_neuronas': mejor_por_accuracy['neuronas'],
        'mejor_accuracy': mejor_por_accuracy['test_accuracy'],
        'mejor_eficiencia': mejor_por_eficiencia['neuronas']
    }
    
    return resultados


def probar_MLP6(X_train, X_test, Y_train, Y_test, arquitecturas=None, epochs_max=30, patience=5):
    print("\n" + "="*60)
    print("TAREA MLP6: Ajustar el número de capas y de neuronas por capa")
    print("="*60)
    
    if arquitecturas is None:
        # MLP6: Diferentes arquitecturas: (lista_neuronas_por_capa, nombre)
        arquitecturas = [
            ([48], "1 capa: 48"),
            ([96], "1 capa: 96"),
            ([48, 48], "2 capas: 48-48"),
            ([96, 48], "2 capas: 96-48"),
            ([48, 48, 24], "3 capas: 48-48-24"),
            ([96, 64, 32], "3 capas: 96-64-32"),
        ]
    
    print(f"\n[MLP6] Arquitecturas a probar: {len(arquitecturas)}")
    for arch in arquitecturas:
        print(f"  - {arch[1]}")
    print(f"[MLP6] Épocas máximas: {epochs_max}")
    print(f"[MLP6] EarlyStopping patience: {patience}")
    
    resultados_arquitecturas = []
    
    for neuronas_capas, nombre in arquitecturas:
        print(f"\n[MLP6] Probando arquitectura: {nombre}...")
        
        # MLP6: Crear modelo con esta arquitectura
        capas = []
        for i, neuronas in enumerate(neuronas_capas):
            if i == 0:
                capas.append(Dense(neuronas, activation='sigmoid', input_shape=(3072,), 
                                  name=f'capa_oculta_{i+1}'))
            else:
                capas.append(Dense(neuronas, activation='sigmoid', name=f'capa_oculta_{i+1}'))
        
        capas.append(Dense(10, activation='softmax', name='capa_salida'))
        model = Sequential(capas)
        
        # MLP6: Compilar modelo
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # MLP6: Crear callback EarlyStopping
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=0
        )
        
        # MLP6: Entrenar modelo
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            validation_split=0.1,
            batch_size=32,
            epochs=epochs_max,
            callbacks=[early_stop],
            verbose=0
        )
        tiempo_entrenamiento = time.time() - inicio
        
        # MLP6: Evaluar con test
        resultados_eval = model.evaluate(X_test, Y_test, verbose=0)
        test_loss = resultados_eval[0]
        test_accuracy = resultados_eval[1]
        
        epocas_entrenadas = len(history.history['loss'])
        mejor_val_acc = max(history.history['val_accuracy'])
        
        # MLP6: Calcular número de parámetros
        num_parametros = model.count_params()
        num_capas = len(neuronas_capas)
        total_neuronas = sum(neuronas_capas)
        
        resultado = {
            'nombre': nombre,
            'neuronas_capas': neuronas_capas,
            'num_capas': num_capas,
            'total_neuronas': total_neuronas,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'tiempo': tiempo_entrenamiento,
            'epocas': epocas_entrenadas,
            'mejor_val_acc': mejor_val_acc,
            'num_parametros': num_parametros,
            'history': history,
            'modelo': model
        }
        
        resultados_arquitecturas.append(resultado)
        
        print(f"  [MLP6] {nombre}: {epocas_entrenadas} épocas, test_acc={test_accuracy:.4f}, "
              f"params={num_parametros}, tiempo={tiempo_entrenamiento:.2f}s")
    
    print("\n[MLP6] Análisis de resultados:")
    print("-" * 90)
    
    print(f"{'Arquitectura':<25} {'Test Acc':<12} {'Parámetros':<15} {'Tiempo (s)':<12} {'Épocas':<10}")
    print("-" * 90)
    for res in resultados_arquitecturas:
        print(f"{res['nombre']:<25} {res['test_accuracy']:<12.4f} "
              f"{res['num_parametros']:<15} {res['tiempo']:<12.2f} {res['epocas']:<10}")
    
    # MLP6: Encontrar mejores arquitecturas
    mejor_por_accuracy = max(resultados_arquitecturas, key=lambda x: x['test_accuracy'])
    mejor_por_eficiencia = min(resultados_arquitecturas, key=lambda x: x['num_parametros'] / x['test_accuracy'])
    
    print("\n[MLP6] Resumen de mejores arquitecturas:")
    print(f"  - Mejor accuracy: {mejor_por_accuracy['nombre']} "
          f"(accuracy={mejor_por_accuracy['test_accuracy']:.4f}, params={mejor_por_accuracy['num_parametros']})")
    print(f"  - Más eficiente (mejor acc/params): {mejor_por_eficiencia['nombre']} "
          f"(accuracy={mejor_por_eficiencia['test_accuracy']:.4f}, params={mejor_por_eficiencia['num_parametros']})")
    
    print("\n[MLP6] Generando gráficas comparativas...")
    
    # MLP6: Gráfica comparativa
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    nombres = [r['nombre'] for r in resultados_arquitecturas]
    accuracies = [r['test_accuracy'] for r in resultados_arquitecturas]
    tiempos = [r['tiempo'] for r in resultados_arquitecturas]
    parametros = [r['num_parametros'] for r in resultados_arquitecturas]
    num_capas_list = [r['num_capas'] for r in resultados_arquitecturas]
    
    # Gráfica de accuracy por arquitectura
    ax1.barh(nombres, accuracies, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax1.set_xlabel('Test Accuracy', fontsize=12)
    ax1.set_ylabel('Arquitectura', fontsize=12)
    ax1.set_title('Test Accuracy por Arquitectura', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Gráfica de parámetros vs accuracy
    scatter = ax2.scatter(parametros, accuracies, s=100, alpha=0.6, c=num_capas_list, cmap='viridis')
    ax2.set_xlabel('Número de Parámetros', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Parámetros (color = num capas)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Número de Capas', fontsize=10)
    for i, nombre in enumerate(nombres):
        ax2.annotate(nombre[:10], (parametros[i], accuracies[i]), fontsize=8, rotation=45)
    
    # Gráfica de tiempo por arquitectura
    ax3.barh(nombres, tiempos, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax3.set_xlabel('Tiempo de Entrenamiento (segundos)', fontsize=12)
    ax3.set_ylabel('Arquitectura', fontsize=12)
    ax3.set_title('Tiempo de Entrenamiento por Arquitectura', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Gráfica de número de capas vs accuracy
    capas_unicas = sorted(set(num_capas_list))
    for capas in capas_unicas:
        indices = [i for i, n in enumerate(num_capas_list) if n == capas]
        accs_capas = [accuracies[i] for i in indices]
        ax4.scatter([capas] * len(accs_capas), accs_capas, s=100, alpha=0.6, 
                   label=f'{capas} capas', edgecolors='black')
    ax4.set_xlabel('Número de Capas', fontsize=12)
    ax4.set_ylabel('Test Accuracy', fontsize=12)
    ax4.set_title('Accuracy vs Número de Capas', fontsize=13, fontweight='bold')
    ax4.set_xticks(capas_unicas)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('MLP6: Comparación de Arquitecturas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # MLP6: Preparar resultados finales
    resultados = {
        'resultados_arquitecturas': resultados_arquitecturas,
        'mejor_arquitectura': mejor_por_accuracy['nombre'],
        'mejor_accuracy': mejor_por_accuracy['test_accuracy'],
        'mejor_eficiencia': mejor_por_eficiencia['nombre']
    }
    
    return resultados


def probar_MLP7(X_train, X_test, Y_train, Y_test, arquitecturas=None, activaciones=None, 
                 batch_sizes=None, epochs_max=50, patience=7, num_repeticiones=3):
    print("\n" + "="*60)
    print("TAREA MLP7: Optimizar la arquitectura de un MLP")
    print("="*60)
    
    # MLP7: Configurar arquitecturas a probar (diferentes capas y neuronas)
    if arquitecturas is None:
        arquitecturas = [
            ([48], "1capa_48"),                    # Arquitectura simple
            ([96], "1capa_96"),                    # Más neuronas en 1 capa
            ([48, 48], "2capas_48-48"),            # 2 capas iguales
            ([96, 48], "2capas_96-48"),            # 2 capas decrecientes
            ([128, 64], "2capas_128-64"),          # 2 capas más grandes
            ([48, 48, 24], "3capas_48-48-24"),     # 3 capas decrecientes
            ([96, 64, 32], "3capas_96-64-32"),     # 3 capas más grandes
            ([128, 96, 64], "3capas_128-96-64"),   # 3 capas grandes
        ]
    
    # MLP7: Configurar funciones de activación a probar
    if activaciones is None:
        activaciones = ['relu', 'sigmoid', 'tanh']  # Las más comunes
    
    # MLP7: Configurar batch sizes a probar
    if batch_sizes is None:
        batch_sizes = [32, 64, 128]  # Valores intermedios eficientes
    
    print(f"\n[MLP7] Configuración de optimización:")
    print(f"  - Arquitecturas: {len(arquitecturas)}")
    print(f"  - Funciones de activación: {activaciones}")
    print(f"  - Batch sizes: {batch_sizes}")
    print(f"  - Épocas máximas: {epochs_max}")
    print(f"  - EarlyStopping patience: {patience}")
    print(f"  - Repeticiones por configuración: {num_repeticiones}")
    print(f"  - Total de configuraciones: {len(arquitecturas) * len(activaciones) * len(batch_sizes)}")
    
    resultados_mlp7 = []
    configuracion_actual = 0
    total_configs = len(arquitecturas) * len(activaciones) * len(batch_sizes)
    
    # MLP7: Bucle principal: probar todas las combinaciones
    for neuronas_capas, nombre_arch in arquitecturas:
        for activacion in activaciones:
            for batch_size in batch_sizes:
                configuracion_actual += 1
                nombre_config = f"{nombre_arch}_{activacion}_batch{batch_size}"
                
                print(f"\n[MLP7] Configuración {configuracion_actual}/{total_configs}: {nombre_config}")
                
                # MLP7: Lista para almacenar resultados de las repeticiones
                historias_config = []
                modelos_config = []
                tiempos_config = []
                test_accs_config = []
                
                # MLP7: Realizar múltiples repeticiones para robustez estadística
                for rep in range(num_repeticiones):
                    print(f"  Repetición {rep+1}/{num_repeticiones}...", end=" ")
                    
                    # MLP7: Construir modelo con esta configuración
                    capas = []
                    for i, neuronas in enumerate(neuronas_capas):
                        if i == 0:
                            # Primera capa oculta: requiere input_shape
                            capas.append(Dense(neuronas, activation=activacion, 
                                             input_shape=(3072,), 
                                             name=f'capa_oculta_{i+1}'))
                        else:
                            # Capas ocultas siguientes
                            capas.append(Dense(neuronas, activation=activacion, 
                                             name=f'capa_oculta_{i+1}'))
                    
                    # MLP7: Capa de salida siempre softmax para clasificación
                    capas.append(Dense(10, activation='softmax', name='capa_salida'))
                    model = Sequential(capas)
                    
                    # MLP7: Compilar modelo con Adam optimizer
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # MLP7: EarlyStopping para evitar sobreentrenamiento y optimizar tiempo
                    early_stop = EarlyStopping(
                        monitor='val_accuracy',     # Monitorear validation accuracy
                        patience=patience,          # Esperar patience épocas sin mejora
                        mode='max',                 # Maximizar la métrica
                        restore_best_weights=True,  # Restaurar mejores pesos al detener
                        verbose=0                   # Silencioso
                    )
                    
                    # MLP7: Entrenar modelo con esta configuración
                    inicio = time.time()
                    history = model.fit(
                        X_train, Y_train,
                        validation_split=0.1,       # 10% para validación
                        batch_size=batch_size,      # Batch size de esta configuración
                        epochs=epochs_max,          # Máximo de épocas
                        callbacks=[early_stop],     # Usar EarlyStopping
                        verbose=0                   # Entrenamiento silencioso
                    )
                    tiempo_entrenamiento = time.time() - inicio
                    
                    # MLP7: Evaluar modelo en conjunto de test
                    resultados_eval = model.evaluate(X_test, Y_test, verbose=0)
                    test_loss = resultados_eval[0]
                    test_accuracy = resultados_eval[1]
                    
                    # MLP7: Guardar resultados de esta repetición
                    historias_config.append(history)
                    modelos_config.append(model)
                    tiempos_config.append(tiempo_entrenamiento)
                    test_accs_config.append(test_accuracy)
                    
                    print(f"test_acc={test_accuracy:.4f}, tiempo={tiempo_entrenamiento:.1f}s")
                
                # MLP7: Promediar resultados de las repeticiones para esta configuración
                tiempo_promedio = np.mean(tiempos_config)
                test_acc_promedio = np.mean(test_accs_config)
                mejor_val_acc_promedio = np.mean([max(h.history['val_accuracy']) 
                                                 for h in historias_config])
                
                # MLP7: Calcular número de parámetros del modelo
                num_parametros = modelos_config[0].count_params()
                
                # MLP7: Obtener mejor modelo de las repeticiones (mayor test accuracy)
                mejor_idx = np.argmax(test_accs_config)
                mejor_modelo = modelos_config[mejor_idx]
                mejor_history = historias_config[mejor_idx]
                epocas_entrenadas = len(mejor_history.history['loss'])
                
                # MLP7: Preparar resultado de esta configuración
                resultado = {
                    'nombre': nombre_config,
                    'arquitectura': nombre_arch,
                    'neuronas_capas': neuronas_capas,
                    'num_capas': len(neuronas_capas),
                    'activacion': activacion,
                    'batch_size': batch_size,
                    'test_accuracy': test_acc_promedio,           # Promediado de repeticiones
                    'test_accuracy_std': np.std(test_accs_config), # Desviación estándar
                    'test_loss': resultados_eval[0],
                    'tiempo': tiempo_promedio,                    # Promediado de repeticiones
                    'epocas': epocas_entrenadas,
                    'mejor_val_acc': mejor_val_acc_promedio,
                    'num_parametros': num_parametros,
                    'mejor_modelo': mejor_modelo,                 # Mejor modelo de repeticiones
                    'mejor_history': mejor_history,
                    'historias': historias_config                 # Todas las historias
                }
                
                resultados_mlp7.append(resultado)
                
                print(f"  [MLP7] {nombre_config}: acc={test_acc_promedio:.4f}±{np.std(test_accs_config):.4f}, "
                      f"params={num_parametros}, tiempo={tiempo_promedio:.1f}s, épocas={epocas_entrenadas}")
    
    # MLP7: Análisis y comparación de resultados
    print("\n" + "="*60)
    print("[MLP7] ANÁLISIS DE RESULTADOS")
    print("="*60)
    
    # MLP7: Ordenar resultados por test accuracy descendente
    resultados_mlp7_ordenados = sorted(resultados_mlp7, 
                                      key=lambda x: x['test_accuracy'], 
                                      reverse=True)
    
    # MLP7: Mostrar tabla de resultados (top 10)
    print("\n[MLP7] Top 10 configuraciones:")
    print("-" * 100)
    print(f"{'#':<4} {'Configuración':<35} {'Test Acc':<12} {'Params':<12} {'Tiempo (s)':<12} {'Épocas':<8}")
    print("-" * 100)
    for i, res in enumerate(resultados_mlp7_ordenados[:10], 1):
        print(f"{i:<4} {res['nombre']:<35} {res['test_accuracy']:<12.4f} "
              f"{res['num_parametros']:<12} {res['tiempo']:<12.1f} {res['epocas']:<8}")
    
    # MLP7: Encontrar mejores configuraciones según diferentes criterios
    mejor_por_accuracy = resultados_mlp7_ordenados[0]  # Ya está ordenado
    mejor_por_eficiencia = min(resultados_mlp7, 
                              key=lambda x: x['num_parametros'] / x['test_accuracy'])
    mejor_por_tiempo = min(resultados_mlp7, key=lambda x: x['tiempo'])
    
    print("\n[MLP7] Resumen de mejores configuraciones:")
    print(f"  - Mejor accuracy: {mejor_por_accuracy['nombre']}")
    print(f"    Accuracy: {mejor_por_accuracy['test_accuracy']:.4f} "
          f"({mejor_por_accuracy['test_accuracy']*100:.2f}%)")
    print(f"    Parámetros: {mejor_por_accuracy['num_parametros']:,}")
    print(f"    Tiempo: {mejor_por_accuracy['tiempo']:.1f}s")
    print(f"    Arquitectura: {mejor_por_accuracy['arquitectura']}")
    print(f"    Activación: {mejor_por_accuracy['activacion']}, "
          f"Batch size: {mejor_por_accuracy['batch_size']}")
    
    print(f"\n  - Más eficiente (mejor acc/params): {mejor_por_eficiencia['nombre']}")
    print(f"    Accuracy: {mejor_por_eficiencia['test_accuracy']:.4f}, "
          f"Params: {mejor_por_eficiencia['num_parametros']:,}, "
          f"Eficiencia: {mejor_por_eficiencia['test_accuracy']/mejor_por_eficiencia['num_parametros']*1000:.6f}")
    
    print(f"\n  - Más rápido: {mejor_por_tiempo['nombre']}")
    print(f"    Tiempo: {mejor_por_tiempo['tiempo']:.1f}s, "
          f"Accuracy: {mejor_por_tiempo['test_accuracy']:.4f}")
    
    # MLP7: Visualización de resultados
    print("\n[MLP7] Generando gráficas comparativas...")
    
    # MLP7: Gráfica 1: Top 10 configuraciones por accuracy
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    top10 = resultados_mlp7_ordenados[:10]
    nombres_top10 = [r['nombre'] for r in top10]
    accs_top10 = [r['test_accuracy'] for r in top10]
    params_top10 = [r['num_parametros'] for r in top10]
    tiempos_top10 = [r['tiempo'] for r in top10]
    
    # MLP7: Subplot 1: Accuracy de top 10
    ax1.barh(nombres_top10, accs_top10, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax1.set_xlabel('Test Accuracy', fontsize=12)
    ax1.set_ylabel('Configuración', fontsize=12)
    ax1.set_title('Top 10: Test Accuracy', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()  # Mejor arriba
    
    # MLP7: Subplot 2: Accuracy vs Parámetros (scatter plot de todas las configuraciones)
    todas_accs = [r['test_accuracy'] for r in resultados_mlp7]
    todas_params = [r['num_parametros'] for r in resultados_mlp7]
    todas_activaciones = [r['activacion'] for r in resultados_mlp7]
    
    # MLP7: Colorear por función de activación
    colores_act = {'relu': 'red', 'sigmoid': 'blue', 'tanh': 'green'}
    for act in set(todas_activaciones):
        indices = [i for i, a in enumerate(todas_activaciones) if a == act]
        ax2.scatter([todas_params[i] for i in indices], 
                   [todas_accs[i] for i in indices],
                   c=colores_act[act], label=act, alpha=0.6, s=50, edgecolors='black')
    ax2.set_xlabel('Número de Parámetros', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Parámetros (color = activación)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MLP7: Subplot 3: Tiempo de entrenamiento (top 10)
    ax3.barh(nombres_top10, tiempos_top10, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax3.set_xlabel('Tiempo de Entrenamiento (segundos)', fontsize=12)
    ax3.set_ylabel('Configuración', fontsize=12)
    ax3.set_title('Top 10: Tiempo de Entrenamiento', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # MLP7: Subplot 4: Número de capas vs Accuracy
    num_capas_todas = [r['num_capas'] for r in resultados_mlp7]
    capas_unicas = sorted(set(num_capas_todas))
    for capas in capas_unicas:
        indices = [i for i, n in enumerate(num_capas_todas) if n == capas]
        accs_capas = [todas_accs[i] for i in indices]
        ax4.scatter([capas] * len(accs_capas), accs_capas, s=100, alpha=0.6, 
                   label=f'{capas} capas', edgecolors='black')
    ax4.set_xlabel('Número de Capas Ocultas', fontsize=12)
    ax4.set_ylabel('Test Accuracy', fontsize=12)
    ax4.set_title('Accuracy vs Número de Capas', fontsize=13, fontweight='bold')
    ax4.set_xticks(capas_unicas)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('MLP7: Comparación de Arquitecturas Optimizadas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # MLP7: Gráfica 2: Evolución del entrenamiento de las mejores configuraciones
    print("\n[MLP7] Generando gráficas de evolución del entrenamiento (top 3)...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    top3 = resultados_mlp7_ordenados[:3]
    
    # MLP7: Subplot 1: Validation Accuracy evolution (top 3)
    for res in top3:
        # MLP7: Promediar historias de las repeticiones para curvas más suaves
        historias_promediadas = promediar_historias(res['historias'])
        ax1.plot(historias_promediadas['val_accuracy'], 
                label=f"{res['nombre']} (acc={res['test_accuracy']:.4f})",
                linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Top 3: Evolución de Validation Accuracy', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MLP7: Subplot 2: Validation Loss evolution (top 3)
    for res in top3:
        historias_promediadas = promediar_historias(res['historias'])
        ax2.plot(historias_promediadas['val_loss'], 
                label=f"{res['nombre']}",
                linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Top 3: Evolución de Validation Loss', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MLP7: Subplot 3: Comparación por función de activación
    activaciones_unicas = sorted(set(todas_activaciones))
    for act in activaciones_unicas:
        indices = [i for i, a in enumerate(todas_activaciones) if a == act]
        accs_act = [todas_accs[i] for i in indices]
        ax3.hist(accs_act, alpha=0.6, label=act, bins=15, edgecolor='black')
    ax3.set_xlabel('Test Accuracy', fontsize=12)
    ax3.set_ylabel('Frecuencia', fontsize=12)
    ax3.set_title('Distribución de Accuracy por Función de Activación', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # MLP7: Subplot 4: Comparación por batch size
    batch_sizes_todos = [r['batch_size'] for r in resultados_mlp7]
    batch_unicos = sorted(set(batch_sizes_todos))
    batch_accs_promedio = []
    for bs in batch_unicos:
        indices = [i for i, b in enumerate(batch_sizes_todos) if b == bs]
        accs_bs = [todas_accs[i] for i in indices]
        batch_accs_promedio.append(np.mean(accs_bs))
    ax4.bar([str(bs) for bs in batch_unicos], batch_accs_promedio, 
           color='skyblue', edgecolor='navy', alpha=0.7)
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('Test Accuracy Promedio', fontsize=12)
    ax4.set_title('Accuracy Promedio por Batch Size', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MLP7: Análisis Detallado de Mejores Configuraciones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # MLP7: Matrices de confusión para las top 3 configuraciones
    print("\n[MLP7] Generando matrices de confusión (top 3 configuraciones)...")
    
    for i, res in enumerate(top3, 1):
        mejor_modelo = res['mejor_modelo']
        y_pred = mejor_modelo.predict(X_test, verbose=0)
        plot_matriz_confusion(Y_test, y_pred, 
                            f"MLP7 - Top {i}: {res['nombre']} (acc={res['test_accuracy']:.4f})")
    
    # MLP7: Tabla resumen final
    print("\n" + "="*60)
    print("[MLP7] TABLA RESUMEN FINAL - Top 5 Configuraciones")
    print("="*60)
    print(f"{'#':<4} {'Configuración':<40} {'Test Acc':<12} {'Params':<15} {'Tiempo':<10} {'Arquitectura':<20}")
    print("-" * 100)
    for i, res in enumerate(resultados_mlp7_ordenados[:5], 1):
        print(f"{i:<4} {res['nombre']:<40} {res['test_accuracy']:<12.4f} "
              f"{res['num_parametros']:<15,} {res['tiempo']:<10.1f} {res['arquitectura']:<20}")
    
    # MLP7: Preparar resultados finales
    resultados = {
        'todos_resultados': resultados_mlp7,
        'mejor_configuracion': mejor_por_accuracy['nombre'],
        'mejor_accuracy': mejor_por_accuracy['test_accuracy'],
        'mejor_modelo': mejor_por_accuracy['mejor_modelo'],
        'mejor_arquitectura': mejor_por_accuracy['arquitectura'],
        'mejor_activacion': mejor_por_accuracy['activacion'],
        'mejor_batch_size': mejor_por_accuracy['batch_size'],
        'mejor_num_parametros': mejor_por_accuracy['num_parametros'],
        'mejor_por_eficiencia': mejor_por_eficiencia['nombre'],
        'mejor_por_tiempo': mejor_por_tiempo['nombre'],
        'top10': resultados_mlp7_ordenados[:10]
    }
    
    return resultados


def mostrar_menu():
    print("\n" + "="*60)
    print("MENÚ PRINCIPAL - Práctica 2: Visión artificial y aprendizaje")
    print("="*60)
    print("\nTareas MLP disponibles:")
    print("  1. MLP1: Definir, entrenar y evaluar un MLP con Keras")
    print("  2. MLP2: Ajustar el valor del parámetro epochs")
    print("  3. MLP3: Ajustar el valor del parámetro batch_size")
    print("  4. MLP4: Probar diferentes funciones de activación")
    print("  5. MLP5: Ajustar el número de neuronas")
    print("  6. MLP6: Ajustar el número de capas y de neuronas por capa")
    print("  7. MLP7: Optimizar la arquitectura de un MLP")
    print("  0. Salir")
    print("\n" + "="*60)
    
    while True:
        try:
            opcion = input("\nSelecciona una opción (0-7): ").strip()
            if opcion in ['0', '1', '2', '3', '4', '5', '6', '7']:
                return int(opcion)
            else:
                print("Opción no válida. Por favor, selecciona 0, 1, 2, 3, 4, 5, 6 o 7.")
        except KeyboardInterrupt:
            print("\n\nOperación cancelada por el usuario.")
            return 0
        except Exception as e:
            print(f"Error: {e}. Por favor, intenta de nuevo.")


if __name__ == "__main__":
    # Cargar y preprocesar datos de CIFAR-10 (una sola vez al inicio)
    print("Cargando y preprocesando datos de CIFAR-10...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    
    # Mostrar menú y ejecutar la tarea seleccionada
    while True:
        opcion = mostrar_menu()
        
        if opcion == 0:
            print("\nSaliendo del programa...")
            break
        
        elif opcion == 1:
            print("\n" + "="*60)
            print("EJECUTANDO TAREA MLP1")
            print("="*60)
            MLP1_model, MLP1_history, MLP1_resultados = probar_MLP1(
                X_train, X_test, Y_train, Y_test
            )
            plot_comparacion_modelos([MLP1_resultados], "Resultados MLP1")
            print("\nTarea MLP1 completada.")
            input("\nPresiona Enter para volver al menú...")
        
        elif opcion == 2:
            print("\n" + "="*60)
            print("EJECUTANDO TAREA MLP2")
            print("="*60)
            
            # MLP2: Permitir al usuario configurar parámetros opcionales
            print("\n[MLP2] Configuración (presiona Enter para valores por defecto):")
            try:
                rep_input = input("  Número de repeticiones [5]: ").strip()
                num_rep = int(rep_input) if rep_input else 5
                
                epochs_input = input("  Épocas máximas [30]: ").strip()
                epochs_max = int(epochs_input) if epochs_input else 30
            except ValueError:
                print("Valores inválidos, usando valores por defecto (5 repeticiones, 30 épocas)")
                num_rep = 5
                epochs_max = 30
            
            MLP2_resultados = probar_MLP2(
                X_train, X_test, Y_train, Y_test,
                num_repeticiones=num_rep,  # MLP2: Número de repeticiones
                epochs_max=epochs_max  # MLP2: Número máximo de épocas para análisis
            )
            
            print("\nTarea MLP2 completada.")
            print(f"[MLP2] Recomendación: Usar {MLP2_resultados['mejor_epoca_acc']} épocas " +
                  f"(mejor val_accuracy: {MLP2_resultados['mejor_val_acc']:.4f})")
            input("\nPresiona Enter para volver al menú...")
        
        elif opcion == 3:
            print("\n" + "="*60)
            print("EJECUTANDO TAREA MLP3")
            print("="*60)
            
            # MLP3: Permitir al usuario configurar parámetros opcionales
            print("\n[MLP3] Configuración (presiona Enter para valores por defecto):")
            try:
                batch_input = input("  Batch sizes a probar (separados por comas) [16,32,64,128,256]: ").strip()
                if batch_input:
                    batch_sizes = [int(x.strip()) for x in batch_input.split(',')]
                else:
                    batch_sizes = [16, 32, 64, 128, 256]
                
                epochs_input = input("  Épocas máximas [50]: ").strip()
                epochs_max = int(epochs_input) if epochs_input else 50
                
                patience_input = input("  Patience para EarlyStopping [5]: ").strip()
                patience = int(patience_input) if patience_input else 5
            except ValueError:
                print("Valores inválidos, usando valores por defecto")
                batch_sizes = [16, 32, 64, 128, 256]
                epochs_max = 50
                patience = 5
            
            MLP3_resultados = probar_MLP3(
                X_train, X_test, Y_train, Y_test,
                batch_sizes=batch_sizes,  # MLP3: Valores de batch_size a probar
                epochs_max=epochs_max,  # MLP3: Número máximo de épocas
                patience=patience  # MLP3: Patience para EarlyStopping
            )
            
            print("\nTarea MLP3 completada.")
            print(f"[MLP3] Recomendación: Batch size óptimo = {MLP3_resultados['mejor_por_eficiencia']} "
                  f"(eficiencia: {MLP3_resultados['mejor_eficiencia']:.6f})")
            print(f"[MLP3] Mejor accuracy: batch_size={MLP3_resultados['mejor_por_accuracy']} "
                  f"(accuracy={MLP3_resultados['mejor_accuracy']:.4f})")
            input("\nPresiona Enter para volver al menú...")
        
        elif opcion == 4:
            print("\n" + "="*60)
            print("EJECUTANDO TAREA MLP4")
            print("="*60)
            
            # MLP4: Permitir al usuario configurar parámetros opcionales
            print("\n[MLP4] Configuración (presiona Enter para valores por defecto):")
            try:
                activaciones_input = input("  Funciones de activación (separadas por comas) [sigmoid,relu,tanh]: ").strip()
                if activaciones_input:
                    activaciones = [x.strip() for x in activaciones_input.split(',')]
                else:
                    activaciones = ['sigmoid', 'relu', 'tanh']
                
                epochs_input = input("  Épocas máximas [30]: ").strip()
                epochs_max = int(epochs_input) if epochs_input else 30
                
                patience_input = input("  Patience para EarlyStopping [5]: ").strip()
                patience = int(patience_input) if patience_input else 5
            except ValueError:
                print("Valores inválidos, usando valores por defecto")
                activaciones = ['sigmoid', 'relu', 'tanh']
                epochs_max = 30
                patience = 5
            
            MLP4_resultados = probar_MLP4(
                X_train, X_test, Y_train, Y_test,
                activaciones=activaciones,
                epochs_max=epochs_max,
                patience=patience
            )
            
            print("\nTarea MLP4 completada.")
            print(f"[MLP4] Mejor activación: {MLP4_resultados['mejor_activacion']} "
                  f"(accuracy={MLP4_resultados['mejor_accuracy']:.4f})")
            input("\nPresiona Enter para volver al menú...")
        
        elif opcion == 5:
            print("\n" + "="*60)
            print("EJECUTANDO TAREA MLP5")
            print("="*60)
            
            # MLP5: Permitir al usuario configurar parámetros opcionales
            print("\n[MLP5] Configuración (presiona Enter para valores por defecto):")
            try:
                neuronas_input = input("  Números de neuronas (separados por comas) [24,48,96,192,384]: ").strip()
                if neuronas_input:
                    num_neuronas = [int(x.strip()) for x in neuronas_input.split(',')]
                else:
                    num_neuronas = [24, 48, 96, 192, 384]
                
                epochs_input = input("  Épocas máximas [30]: ").strip()
                epochs_max = int(epochs_input) if epochs_input else 30
                
                patience_input = input("  Patience para EarlyStopping [5]: ").strip()
                patience = int(patience_input) if patience_input else 5
            except ValueError:
                print("Valores inválidos, usando valores por defecto")
                num_neuronas = [24, 48, 96, 192, 384]
                epochs_max = 30
                patience = 5
            
            MLP5_resultados = probar_MLP5(
                X_train, X_test, Y_train, Y_test,
                num_neuronas=num_neuronas,
                epochs_max=epochs_max,
                patience=patience
            )
            
            print("\nTarea MLP5 completada.")
            print(f"[MLP5] Mejor número de neuronas: {MLP5_resultados['mejor_neuronas']} "
                  f"(accuracy={MLP5_resultados['mejor_accuracy']:.4f})")
            print(f"[MLP5] Mejor eficiencia: {MLP5_resultados['mejor_eficiencia']} neuronas")
            input("\nPresiona Enter para volver al menú...")
        
        elif opcion == 6:
            print("\n" + "="*60)
            print("EJECUTANDO TAREA MLP6")
            print("="*60)
            
            # MLP6: Permitir al usuario configurar parámetros opcionales
            print("\n[MLP6] Configuración (presiona Enter para valores por defecto):")
            print("  [MLP6] Usando arquitecturas por defecto (configurables en el código)")
            
            epochs_input = input("  Épocas máximas [30]: ").strip()
            epochs_max = int(epochs_input) if epochs_input else 30
            
            patience_input = input("  Patience para EarlyStopping [5]: ").strip()
            patience = int(patience_input) if patience_input else 5
            
            MLP6_resultados = probar_MLP6(
                X_train, X_test, Y_train, Y_test,
                arquitecturas=None,  # Usa valores por defecto
                epochs_max=epochs_max,
                patience=patience
            )
            
            print("\nTarea MLP6 completada.")
            print(f"[MLP6] Mejor arquitectura: {MLP6_resultados['mejor_arquitectura']} "
                  f"(accuracy={MLP6_resultados['mejor_accuracy']:.4f})")
            print(f"[MLP6] Arquitectura más eficiente: {MLP6_resultados['mejor_eficiencia']}")
            input("\nPresiona Enter para volver al menú...")
        
        elif opcion == 7:
            print("\n" + "="*60)
            print("EJECUTANDO TAREA MLP7")
            print("="*60)
            print("\n[MLP7] ADVERTENCIA: Esta tarea probará múltiples configuraciones y puede tardar mucho tiempo.")
            print("[MLP7] Se recomienda usar valores por defecto para una primera ejecución.")
            
            # MLP7: Permitir al usuario configurar parámetros opcionales
            print("\n[MLP7] Configuración (presiona Enter para valores por defecto):")
            
            continuar = input("\n  ¿Deseas continuar? (s/n) [s]: ").strip().lower()
            if continuar and continuar != 's':
                print("[MLP7] Operación cancelada.")
                input("\nPresiona Enter para volver al menú...")
                continue
            
            try:
                epochs_input = input("  Épocas máximas [50]: ").strip()
                epochs_max = int(epochs_input) if epochs_input else 50
                
                patience_input = input("  Patience para EarlyStopping [7]: ").strip()
                patience = int(patience_input) if patience_input else 7
                
                rep_input = input("  Repeticiones por configuración [3]: ").strip()
                num_repeticiones = int(rep_input) if rep_input else 3
                
                print("\n[MLP7] Usando configuraciones por defecto:")
                print("  - Arquitecturas: 8 configuraciones variadas")
                print("  - Activaciones: ['relu', 'sigmoid', 'tanh']")
                print("  - Batch sizes: [32, 64, 128]")
                print(f"  - Total: 8 x 3 x 3 = 72 configuraciones x {num_repeticiones} repeticiones = {72 * num_repeticiones} entrenamientos")
            except ValueError:
                print("Valores inválidos, usando valores por defecto")
                epochs_max = 50
                patience = 7
                num_repeticiones = 3
            
            confirmar = input("\n[MLP7] ¿Confirmar y comenzar? (s/n) [s]: ").strip().lower()
            if confirmar and confirmar != 's':
                print("[MLP7] Operación cancelada.")
                input("\nPresiona Enter para volver al menú...")
                continue
            
            MLP7_resultados = probar_MLP7(
                X_train, X_test, Y_train, Y_test,
                arquitecturas=None,      # Usa valores por defecto
                activaciones=None,       # Usa valores por defecto
                batch_sizes=None,        # Usa valores por defecto
                epochs_max=epochs_max,   # MLP7: Épocas máximas
                patience=patience,       # MLP7: Patience para EarlyStopping
                num_repeticiones=num_repeticiones  # MLP7: Repeticiones por configuración
            )
            
            print("\n" + "="*60)
            print("[MLP7] OPTIMIZACIÓN COMPLETADA")
            print("="*60)
            print(f"\n[MLP7] Mejor configuración: {MLP7_resultados['mejor_configuracion']}")
            print(f"[MLP7] Mejor test accuracy: {MLP7_resultados['mejor_accuracy']:.4f} ({MLP7_resultados['mejor_accuracy']*100:.2f}%)")
            print(f"[MLP7] Arquitectura: {MLP7_resultados['mejor_arquitectura']}")
            print(f"[MLP7] Función de activación: {MLP7_resultados['mejor_activacion']}")
            print(f"[MLP7] Batch size: {MLP7_resultados['mejor_batch_size']}")
            print(f"[MLP7] Número de parámetros: {MLP7_resultados['mejor_num_parametros']:,}")
            print(f"\n[MLP7] Configuración más eficiente: {MLP7_resultados['mejor_por_eficiencia']}")
            print(f"[MLP7] Configuración más rápida: {MLP7_resultados['mejor_por_tiempo']}")
            input("\nPresiona Enter para volver al menú...")

