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
    """
    Carga los datos de CIFAR-10 y los preprocesa para MLP.
    
    Esta función es utilizada por MLP1 (y otras tareas MLP) para preparar
    los datos según los requisitos del enunciado:
    - Para MLP, las imágenes deben ser aplanadas en un vector unidimensional
    - Normaliza los valores de píxeles al rango [0, 1]
    - Convierte las etiquetas a formato categórico one-hot (necesario para
      categorical_crossentropy usado en MLP1)
    
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test) preprocesados
    """
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
    """
    Muestra gráficas de la evolución de la pérdida y la tasa de acierto
    durante el entrenamiento para el conjunto de entrenamiento y validación.
    
    Args:
        history: Objeto History devuelto por model.fit()
        titulo: Título de la gráfica
    """
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
    """
    Muestra una gráfica de barras comparando el tiempo de entrenamiento
    y la tasa de acierto final de varios modelos.
    
    Args:
        resultados: Lista de diccionarios con keys 'nombre', 'tiempo', 'accuracy'
        titulo: Título de la gráfica
    """
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
    """
    Muestra la matriz de confusión para los resultados de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas (formato one-hot o índices)
        y_pred: Predicciones del modelo (formato one-hot o índices)
        titulo: Título de la gráfica
    """
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
    """
    ============================================================================
    TAREA MLP1: Definir, entrenar y evaluar un MLP con Keras
    ============================================================================
    
    Esta función implementa la Tarea MLP1 según el enunciado de la práctica.
    Define un Perceptrón Multicapa (MLP) con las siguientes especificaciones:
    
    Especificaciones del MLP1 (según enunciado):
    - Capa oculta Dense: 48 neuronas, activación sigmoid
    - Capa de salida Dense: 10 neuronas, activación softmax
    - Optimizador: Adam
    - Loss: categorical_crossentropy
    - Metrics: accuracy
    - Parámetros de entrenamiento: validation_split=0.1, batch_size=32, epochs=10
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de test
        Y_train: Etiquetas de entrenamiento (one-hot)
        Y_test: Etiquetas de test (one-hot)
    
    Returns:
        model: Modelo entrenado
        history: Historial de entrenamiento
        resultados: Diccionario con resultados de evaluación
    """
    print("\n" + "="*60)
    print("TAREA MLP1: Definir, entrenar y evaluar un MLP con Keras")
    print("="*60)
    
    # ========================================================================
    # MLP1 - PASO 1: DEFINIR EL MODELO
    # ========================================================================
    # Según el enunciado, el MLP1 debe tener:
    # - Una capa oculta Dense con 48 neuronas y función de activación sigmoid
    # - Una capa de salida Dense con 10 neuronas (10 clases de CIFAR-10) 
    #   y función de activación softmax
    # ========================================================================
    model = Sequential([
        Dense(48, activation='sigmoid', input_shape=(3072,), name='capa_oculta'),  # MLP1: 48 neuronas, sigmoid
        Dense(10, activation='softmax', name='capa_salida')  # MLP1: 10 neuronas, softmax
    ])
    
    # ========================================================================
    # MLP1 - PASO 2: COMPILAR EL MODELO
    # ========================================================================
    # Según el enunciado, el MLP1 debe usar:
    # - Optimizador: Adam
    # - Loss: categorical_crossentropy
    # - Metrics: accuracy
    # ========================================================================
    model.compile(
        optimizer='adam',  # MLP1: Optimizador Adam (requisito del enunciado)
        loss='categorical_crossentropy',  # MLP1: Loss categorical_crossentropy (requisito del enunciado)
        metrics=['accuracy']  # MLP1: Métrica accuracy (requisito del enunciado)
    )
    
    # ========================================================================
    # MLP1 - PASO 3: MOSTRAR RESUMEN DEL MODELO
    # ========================================================================
    # El enunciado requiere mostrar un resumen de la estructura del modelo
    # ========================================================================
    print("\nResumen del modelo:")
    model.summary()
    
    # ========================================================================
    # MLP1 - PASO 4: ENTRENAR EL MODELO
    # ========================================================================
    # Según el enunciado, el MLP1 debe entrenarse con:
    # - validation_split=0.1 (10% de los datos de entrenamiento para validación)
    # - batch_size=32
    # - epochs=10
    # ========================================================================
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
    
    # ========================================================================
    # MLP1 - PASO 5: EVALUAR EL MODELO CON EL CONJUNTO DE TEST
    # ========================================================================
    # El enunciado requiere evaluar el modelo con el conjunto de test
    # usando la función evaluate() de Keras
    # ========================================================================
    print("\nEvaluando modelo con conjunto de test...")
    resultados_eval = model.evaluate(X_test, Y_test, verbose=1)
    test_loss = resultados_eval[0]
    test_accuracy = resultados_eval[1]
    
    print(f"\nResultados en conjunto de test:")
    print(f"  Pérdida: {test_loss:.4f}")
    print(f"  Tasa de acierto: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # ========================================================================
    # MLP1 - PASO 6: OBTENER PREDICCIONES Y VISUALIZAR RESULTADOS
    # ========================================================================
    # El enunciado requiere mostrar gráficas de evolución del entrenamiento
    # y matriz de confusión para analizar los resultados del MLP1
    # ========================================================================
    # Obtener predicciones para la matriz de confusión
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
    """
    ============================================================================
    MLP2 - FUNCIÓN AUXILIAR: Promediar múltiples historiales de entrenamiento
    ============================================================================
    
    Esta función promedia los resultados de múltiples entrenamientos del mismo
    modelo para obtener resultados más robustos estadísticamente.
    Utilizada en MLP2 para promediar 5 repeticiones del entrenamiento.
    
    Args:
        historias: Lista de objetos History devueltos por model.fit()
    
    Returns:
        dict: Diccionario con las métricas promediadas
    """
    # MLP2: Inicializar diccionario para almacenar promedios
    promedios = {}
    
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
    """
    ============================================================================
    MLP2 - FUNCIÓN DE VISUALIZACIÓN: Gráficas de evolución con ejes Y separados
    ============================================================================
    
    Esta función muestra las curvas de entrenamiento y validación para MLP2.
    Utiliza ejes Y separados para accuracy y loss, permitiendo ajustar los
    rangos independientemente y apreciar mejor la pendiente de las curvas.
    
    Args:
        historias_promediadas: Diccionario con métricas promediadas
        titulo: Título de la gráfica
    """
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
    """
    ============================================================================
    TAREA MLP2: Ajustar el valor del parámetro epochs
    ============================================================================
    
    Esta función implementa la Tarea MLP2 según el enunciado de la práctica.
    
    Objetivos de MLP2:
    1. Analizar la evolución del entrenamiento para detectar:
       - Entrenamiento insuficiente (podría mejorar con más épocas)
       - Sobreentrenamiento (el modelo se ajusta demasiado y pierde generalidad)
    2. Realizar múltiples repeticiones (5) para obtener resultados robustos
    3. Ajustar el número de épocas basándose en las gráficas
    4. Implementar EarlyStopping para detección automática de sobreentrenamiento
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de test
        Y_train: Etiquetas de entrenamiento (one-hot)
        Y_test: Etiquetas de test (one-hot)
        num_repeticiones: Número de repeticiones del entrenamiento (default: 5)
        epochs_max: Número máximo de épocas para el entrenamiento (default: 30)
    
    Returns:
        dict: Diccionario con resultados y recomendaciones
    """
    print("\n" + "="*60)
    print("TAREA MLP2: Ajustar el valor del parámetro epochs")
    print("="*60)
    
    # ========================================================================
    # MLP2 - PASO 1: DEFINIR EL MODELO (MISMO QUE MLP1)
    # ========================================================================
    # MLP2 usa la misma arquitectura que MLP1 para analizar el efecto de
    # diferentes números de épocas en el mismo modelo
    # ========================================================================
    print("\n[MLP2] Modelo: Misma arquitectura que MLP1 (48 neuronas sigmoid, 10 softmax)")
    
    # ========================================================================
    # MLP2 - PASO 2: ENTRENAR CON MÚLTIPLES REPETICIONES
    # ========================================================================
    # Según el enunciado, es necesario realizar 5 repeticiones independientes
    # del mismo entrenamiento para obtener resultados robustos estadísticamente.
    # Cada repetición parte de estados iniciales aleatorios diferentes.
    # ========================================================================
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
    
    # ========================================================================
    # MLP2 - PASO 3: PROMEDIAR RESULTADOS
    # ========================================================================
    # Según el enunciado, debemos promediar los resultados de las 5 repeticiones
    # para obtener curvas más suaves y representativas
    # ========================================================================
    print("\n[MLP2] Promediando resultados de las repeticiones...")
    historias_promediadas = promediar_historias(historias)
    
    # ========================================================================
    # MLP2 - PASO 4: VISUALIZAR EVOLUCIÓN DEL ENTRENAMIENTO
    # ========================================================================
    # El enunciado requiere visualizar las curvas de train_accuracy, train_loss,
    # validation_accuracy y validation_loss para detectar sobreentrenamiento
    # ========================================================================
    print("\n[MLP2] Generando gráficas de evolución del entrenamiento...")
    plot_evolucion_mlp2(historias_promediadas, 
                        f"MLP2: Evolución del Entrenamiento ({num_repeticiones} repeticiones promediadas)")
    
    # ========================================================================
    # MLP2 - PASO 5: ANALIZAR Y DETERMINAR ÉPOCAS ÓPTIMAS
    # ========================================================================
    # Analizar las curvas para determinar:
    # - Si el entrenamiento se detuvo prematuramente
    # - Si hubo sobreentrenamiento (curvas de validación empeoran)
    # - El número óptimo de épocas basado en val_accuracy o val_loss
    # ========================================================================
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
            print(f"  - ⚠️  Sobreentrenamiento detectado: val_loss empeora después de época {mejor_epoca_loss + 1}")
        else:
            print(f"  - ✓ No se detectó sobreentrenamiento significativo")
    
    # ========================================================================
    # MLP2 - PASO 6: IMPLEMENTAR EARLYSTOPPING
    # ========================================================================
    # Según el enunciado, una vez comprendido el ajuste manual, debemos
    # implementar EarlyStopping para detección automática de sobreentrenamiento
    # ========================================================================
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


def mostrar_menu():
    """
    ============================================================================
    MENÚ PRINCIPAL - Selección de tareas MLP
    ============================================================================
    
    Muestra un menú interactivo para seleccionar qué tarea MLP ejecutar.
    """
    print("\n" + "="*60)
    print("MENÚ PRINCIPAL - Práctica 2: Visión artificial y aprendizaje")
    print("="*60)
    print("\nTareas MLP disponibles:")
    print("  1. MLP1: Definir, entrenar y evaluar un MLP con Keras")
    print("  2. MLP2: Ajustar el valor del parámetro epochs")
    print("  0. Salir")
    print("\n" + "="*60)
    
    while True:
        try:
            opcion = input("\nSelecciona una opción (0-2): ").strip()
            if opcion in ['0', '1', '2']:
                return int(opcion)
            else:
                print("⚠️  Opción no válida. Por favor, selecciona 0, 1 o 2.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operación cancelada por el usuario.")
            return 0
        except Exception as e:
            print(f"⚠️  Error: {e}. Por favor, intenta de nuevo.")


if __name__ == "__main__":
    # ========================================================================
    # EJECUCIÓN PRINCIPAL - MENÚ DE SELECCIÓN DE TAREAS MLP
    # ========================================================================
    # Este bloque muestra un menú interactivo para seleccionar qué tarea MLP
    # ejecutar. Según el enunciado, solo se debe ejecutar una tarea a la vez.
    # ========================================================================
    
    # Cargar y preprocesar datos de CIFAR-10 (una sola vez al inicio)
    print("Cargando y preprocesando datos de CIFAR-10...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    
    # Mostrar menú y ejecutar la tarea seleccionada
    while True:
        opcion = mostrar_menu()
        
        if opcion == 0:
            print("\n👋 Saliendo del programa...")
            break
        
        elif opcion == 1:
            # ========================================================================
            # EJECUTAR TAREA MLP1
            # ========================================================================
            # Ejecuta la función probar_MLP1() que implementa todos los requisitos
            # de la Tarea MLP1 según el enunciado:
            # - Define un MLP con 48 neuronas ocultas (sigmoid) y 10 de salida (softmax)
            # - Compila con Adam, categorical_crossentropy y accuracy
            # - Entrena con validation_split=0.1, batch_size=32, epochs=10
            # - Evalúa con el conjunto de test
            # - Muestra gráficas de evolución y matriz de confusión
            # ========================================================================
            print("\n" + "="*60)
            print("EJECUTANDO TAREA MLP1")
            print("="*60)
            MLP1_model, MLP1_history, MLP1_resultados = probar_MLP1(
                X_train, X_test, Y_train, Y_test
            )
            plot_comparacion_modelos([MLP1_resultados], "Resultados MLP1")
            print("\n✅ Tarea MLP1 completada.")
            input("\nPresiona Enter para volver al menú...")
        
        elif opcion == 2:
            # ========================================================================
            # EJECUTAR TAREA MLP2
            # ========================================================================
            # Ejecuta la función probar_MLP2() que implementa todos los requisitos
            # de la Tarea MLP2 según el enunciado:
            # - Analiza la evolución del entrenamiento para detectar sobreentrenamiento
            # - Realiza 5 repeticiones del entrenamiento para robustez estadística
            # - Promedia los resultados de las repeticiones
            # - Visualiza curvas de entrenamiento y validación con ejes Y separados
            # - Determina el número óptimo de épocas basándose en val_accuracy/val_loss
            # - Implementa EarlyStopping con diferentes configuraciones
            # ========================================================================
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
                print("  ⚠️  Valores inválidos, usando valores por defecto (5 repeticiones, 30 épocas)")
                num_rep = 5
                epochs_max = 30
            
            MLP2_resultados = probar_MLP2(
                X_train, X_test, Y_train, Y_test,
                num_repeticiones=num_rep,  # MLP2: Número de repeticiones
                epochs_max=epochs_max  # MLP2: Número máximo de épocas para análisis
            )
            
            print("\n✅ Tarea MLP2 completada.")
            print(f"[MLP2] Recomendación: Usar {MLP2_resultados['mejor_epoca_acc']} épocas " +
                  f"(mejor val_accuracy: {MLP2_resultados['mejor_val_acc']:.4f})")
            input("\nPresiona Enter para volver al menú...")

