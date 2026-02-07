"""
Configuración de Machine Learning
"""

import os
import numpy as np


class MLConfig:
    """Configuración de modelos de ML"""
    
    # Rutas de modelos
    BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/trained_models')
    
    LSC_MODEL_PATH = os.path.join(BASE_MODEL_DIR, 'lsc_model.h5')
    ASL_MODEL_PATH = os.path.join(BASE_MODEL_DIR, 'asl_model.h5')
    
    # Configuración de MediaPipe
    MEDIAPIPE_CONFIG = {
        'static_image_mode': False,
        'max_num_hands': 2,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.5,
        'model_complexity': 1  # 0=lite, 1=full
    }
    
    # Configuración de predicción
    PREDICTION_THRESHOLD = 0.6  # Confianza mínima para aceptar predicción
    SMOOTHING_WINDOW = 5  # Suavizado de predicciones
    
    # Preprocesamiento
    INPUT_SHAPE = (63,)  # 21 landmarks * 3 coordenadas
    NUM_CLASSES_LSC = 30  # Número de clases LSC
    NUM_CLASSES_ASL = 26  # Número de clases ASL (A-Z)
    
    # Normalización
    NORMALIZE = True
    NORMALIZATION_MEAN = 0.5
    NORMALIZATION_STD = 0.5
    
    # Diccionarios de señas
    LSC_LABELS = {
        0: 'Hola', 1: 'Gracias', 2: 'Por favor', 3: 'Sí', 4: 'No',
        5: 'Ayuda', 6: 'Comer', 7: 'Beber', 8: 'Agua', 9: 'Baño',
        10: 'Casa', 11: 'Familia', 12: 'Amigo', 13: 'Trabajo', 14: 'Estudiar',
        15: 'Leer', 16: 'Escribir', 17: 'Hablar', 18: 'Escuchar', 19: 'Ver',
        20: 'Amor', 21: 'Feliz', 22: 'Triste', 23: 'Enojado', 24: 'Cansado',
        25: 'Dormir', 26: 'Despertar', 27: 'Día', 28: 'Noche', 29: 'Tiempo'
    }
    
    ASL_LABELS = {
        i: chr(65 + i) for i in range(26)  # A-Z
    }
    
    # Configuración de TensorFlow
    TF_CONFIG = {
        'inter_op_parallelism_threads': 2,
        'intra_op_parallelism_threads': 2,
        'allow_growth': True  # Permitir crecimiento de memoria GPU
    }
    
    # Data augmentation (para entrenamiento)
    AUGMENTATION_CONFIG = {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
    
    # Configuración de entrenamiento
    TRAINING_CONFIG = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy']
    }
    
    @staticmethod
    def get_model_path(language='LSC'):
        """Obtener ruta del modelo según idioma"""
        if language.upper() == 'LSC':
            return MLConfig.LSC_MODEL_PATH
        elif language.upper() == 'ASL':
            return MLConfig.ASL_MODEL_PATH
        else:
            raise ValueError(f"Idioma no soportado: {language}")
    
    @staticmethod
    def get_labels(language='LSC'):
        """Obtener etiquetas según idioma"""
        if language.upper() == 'LSC':
            return MLConfig.LSC_LABELS
        elif language.upper() == 'ASL':
            return MLConfig.ASL_LABELS
        else:
            raise ValueError(f"Idioma no soportado: {language}")
    
    @staticmethod
    def normalize_landmarks(landmarks):
        """Normalizar landmarks"""
        if MLConfig.NORMALIZE:
            landmarks = (landmarks - MLConfig.NORMALIZATION_MEAN) / MLConfig.NORMALIZATION_STD
        return landmarks
    
    @staticmethod
    def configure_tensorflow():
        """Configurar TensorFlow"""
        import tensorflow as tf
        
        # Configurar threads
        tf.config.threading.set_inter_op_parallelism_threads(
            MLConfig.TF_CONFIG['inter_op_parallelism_threads']
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            MLConfig.TF_CONFIG['intra_op_parallelism_threads']
        )
        
        # Configurar GPU (si está disponible)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(
                        gpu, 
                        MLConfig.TF_CONFIG['allow_growth']
                    )
                print(f"✅ GPU detectada: {len(gpus)} dispositivo(s)")
            except RuntimeError as e:
                print(f"⚠️ Error configurando GPU: {e}")
        else:
            print("ℹ️ No se detectó GPU, usando CPU")