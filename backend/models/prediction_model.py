"""
Modelo de predicción con entrenamiento y evaluación
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List, Dict
import logging
import json
import os

logger = logging.getLogger(__name__)


class PredictionModel:
    """
    Modelo de red neuronal para clasificación de señas
    """
    
    def __init__(
        self,
        input_shape: int = 63,
        num_classes: int = 30,
        model_path: str = None
    ):
        """
        Inicializa el modelo
        
        Args:
            input_shape: Tamaño del input (21 landmarks * 3)
            num_classes: Número de clases
            model_path: Ruta para cargar modelo existente
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            logger.info(f"Modelo cargado desde {model_path}")
        else:
            self.model = self._build_model()
            logger.info("Modelo nuevo creado")
        
        self.history = None
    
    def _build_model(self) -> keras.Model:
        """
        Construye la arquitectura del modelo
        
        Returns:
            Modelo de Keras
        """
        model = keras.Sequential([
            # Capa de entrada
            layers.Input(shape=(self.input_shape,)),
            
            # Capas densas
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Capa de salida
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        optimizer: str = 'adam'
    ):
        """
        Compila el modelo
        
        Args:
            learning_rate: Tasa de aprendizaje
            optimizer: Optimizador a usar
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Modelo compilado con {optimizer}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: List = None
    ) -> keras.callbacks.History:
        """
        Entrena el modelo
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación
            y_val: Etiquetas de validación
            epochs: Número de épocas
            batch_size: Tamaño del batch
            callbacks: Callbacks de Keras
            
        Returns:
            Historia del entrenamiento
        """
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        logger.info(f"Iniciando entrenamiento: {epochs} épocas")
        
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Entrenamiento completado")
        
        return self.history
    
    def _get_default_callbacks(self) -> List:
        """Retorna callbacks por defecto"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        if self.model_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    self.model_path,
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            )
        
        return callbacks
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evalúa el modelo
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas de prueba
            
        Returns:
            Tuple de (loss, accuracy)
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"Evaluación - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def predict(
        self,
        X: np.ndarray,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Realiza predicciones
        
        Args:
            X: Datos de entrada
            return_probs: Si True, retorna probabilidades
            
        Returns:
            Predicciones o probabilidades
        """
        predictions = self.model.predict(X, verbose=0)
        
        if return_probs:
            return predictions
        else:
            return np.argmax(predictions, axis=1)
    
    def save(self, path: str = None):
        """
        Guarda el modelo
        
        Args:
            path: Ruta donde guardar (usa self.model_path si None)
        """
        save_path = path or self.model_path
        
        if save_path is None:
            logger.error("No se especificó ruta para guardar")
            return
        
        # Guardar modelo
        self.model.save(save_path)
        
        # Guardar metadata
        metadata = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_path': save_path
        }
        
        metadata_path = save_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Modelo guardado en {save_path}")
    
    def get_summary(self) -> str:
        """Retorna resumen del modelo"""
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def plot_history(self, save_path: str = None):
        """
        Grafica la historia del entrenamiento
        
        Args:
            save_path: Ruta para guardar la gráfica
        """
        if self.history is None:
            logger.warning("No hay historial de entrenamiento")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Model Loss')
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.set_title('Model Accuracy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Gráfica guardada en {save_path}")
        else:
            plt.show()