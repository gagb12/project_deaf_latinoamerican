"""
Detector principal de lenguaje de señas
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SignLanguageDetector:
    """
    Detector de lenguaje de señas usando MediaPipe y TensorFlow
    """
    
    def __init__(self, model_path: str, language: str = 'LSC'):
        """
        Inicializa el detector
        
        Args:
            model_path: Ruta al modelo entrenado
            language: Idioma (LSC o ASL)
        """
        self.language = language.upper()
        self.model_path = model_path
        
        # Cargar modelo
        try:
            self.model = load_model(model_path)
            logger.info(f"✅ Modelo {language} cargado desde {model_path}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            self.model = None
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Labels según idioma
        self.labels = self._load_labels()
        
        # Buffer para suavizado de predicciones
        self.prediction_buffer = []
        self.buffer_size = 5
        
        # Estadísticas
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_confidence': 0.0
        }
    
    def _load_labels(self) -> Dict[int, str]:
        """Carga las etiquetas según el idioma"""
        if self.language == 'LSC':
            return {
                0: 'Hola', 1: 'Gracias', 2: 'Por favor', 3: 'Sí', 4: 'No',
                5: 'Ayuda', 6: 'Comer', 7: 'Beber', 8: 'Agua', 9: 'Baño',
                10: 'Casa', 11: 'Familia', 12: 'Amigo', 13: 'Trabajo', 14: 'Estudiar',
                15: 'Leer', 16: 'Escribir', 17: 'Hablar', 18: 'Escuchar', 19: 'Ver',
                20: 'Amor', 21: 'Feliz', 22: 'Triste', 23: 'Enojado', 24: 'Cansado',
                25: 'Dormir', 26: 'Despertar', 27: 'Día', 28: 'Noche', 29: 'Tiempo'
            }
        else:  # ASL
            return {i: chr(65 + i) for i in range(26)}  # A-Z
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detecta manos en el frame
        
        Args:
            frame: Frame de video (BGR)
            
        Returns:
            Tuple de (frame con landmarks dibujados, landmarks normalizados)
        """
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Procesar
        results = self.hands.process(image_rgb)
        
        # Convertir de vuelta a BGR
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        landmarks = None
        
        if results.multi_hand_landmarks:
            # Dibujar landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Extraer landmarks
            landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
        
        return frame, landmarks
    
    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extrae y normaliza landmarks de la mano"""
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normaliza landmarks"""
        # Normalizar por muñeca (landmark 0)
        wrist = landmarks[:3]
        normalized = landmarks.reshape(-1, 3) - wrist
        
        # Escalar
        max_dist = np.max(np.abs(normalized))
        if max_dist > 0:
            normalized = normalized / max_dist
        
        return normalized.flatten()
    
    def predict(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Realiza predicción desde un frame
        
        Args:
            frame: Frame de video
            
        Returns:
            Dict con predicción o None si no hay detección
        """
        if self.model is None:
            logger.error("Modelo no cargado")
            return None
        
        # Detectar manos
        processed_frame, landmarks = self.detect_hands(frame)
        
        if landmarks is None:
            self.stats['failed_predictions'] += 1
            return None
        
        # Normalizar
        normalized_landmarks = self._normalize_landmarks(landmarks)
        
        # Reshape para predicción
        input_data = normalized_landmarks.reshape(1, -1)
        
        # Predecir
        try:
            prediction = self.model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction[0])
            confidence = float(prediction[0][class_id])
            
            # Actualizar buffer para suavizado
            self._update_prediction_buffer(class_id, confidence)
            
            # Obtener predicción suavizada
            smoothed_class_id, smoothed_confidence = self._get_smoothed_prediction()
            
            # Actualizar estadísticas
            self.stats['total_predictions'] += 1
            self.stats['successful_predictions'] += 1
            self._update_average_confidence(smoothed_confidence)
            
            result = {
                'text': self.labels.get(smoothed_class_id, 'Desconocido'),
                'confidence': smoothed_confidence,
                'class_id': smoothed_class_id,
                'language': self.language,
                'raw_confidence': confidence,
                'raw_class_id': class_id
            }
            
            logger.debug(f"Predicción: {result['text']} ({result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            self.stats['failed_predictions'] += 1
            return None
    
    def _update_prediction_buffer(self, class_id: int, confidence: float):
        """Actualiza buffer de predicciones para suavizado"""
        self.prediction_buffer.append((class_id, confidence))
        
        # Mantener tamaño del buffer
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
    
    def _get_smoothed_prediction(self) -> Tuple[int, float]:
        """Obtiene predicción suavizada del buffer"""
        if not self.prediction_buffer:
            return 0, 0.0
        
        # Contar ocurrencias de cada clase
        class_counts = {}
        confidence_sum = {}
        
        for class_id, confidence in self.prediction_buffer:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            confidence_sum[class_id] = confidence_sum.get(class_id, 0.0) + confidence
        
        # Clase más frecuente
        most_common_class = max(class_counts, key=class_counts.get)
        
        # Confianza promedio de esa clase
        avg_confidence = confidence_sum[most_common_class] / class_counts[most_common_class]
        
        return most_common_class, avg_confidence
    
    def _update_average_confidence(self, confidence: float):
        """Actualiza confianza promedio"""
        total = self.stats['total_predictions']
        current_avg = self.stats['average_confidence']
        
        self.stats['average_confidence'] = (
            (current_avg * (total - 1) + confidence) / total
        )
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del detector"""
        success_rate = 0.0
        if self.stats['total_predictions'] > 0:
            success_rate = (
                self.stats['successful_predictions'] / 
                self.stats['total_predictions']
            )
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'language': self.language,
            'model_loaded': self.model is not None
        }
    
    def reset_stats(self):
        """Reinicia estadísticas"""
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_confidence': 0.0
        }
        self.prediction_buffer = []
    
    def __del__(self):
        """Limpieza al destruir objeto"""
        if hasattr(self, 'hands'):
            self.hands.close()