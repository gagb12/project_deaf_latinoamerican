"""
Rastreador de manos usando MediaPipe
"""

import mediapipe as mp
import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HandTracker:
    """
    Rastreador de manos con MediaPipe
    """
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Inicializa el rastreador
        
        Args:
            max_num_hands: Número máximo de manos a detectar
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para seguimiento
            model_complexity: Complejidad del modelo (0=lite, 1=full)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
        logger.info("HandTracker inicializado")
    
    def process_frame(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = True
    ) -> Tuple[np.ndarray, List]:
        """
        Procesa un frame y detecta manos
        
        Args:
            frame: Frame BGR
            draw_landmarks: Si True, dibuja landmarks
            
        Returns:
            Tuple de (frame procesado, lista de manos detectadas)
        """
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Procesar
        results = self.hands.process(image_rgb)
        
        # Convertir de vuelta
        image_rgb.flags.writeable = True
        processed_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        hands_data = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Dibujar si se requiere
                if draw_landmarks:
                    self._draw_hand_landmarks(processed_frame, hand_landmarks)
                
                # Extraer datos
                hand_data = {
                    'landmarks': self._extract_landmarks(hand_landmarks),
                    'handedness': handedness.classification[0].label,
                    'score': handedness.classification[0].score
                }
                
                hands_data.append(hand_data)
        
        return processed_frame, hands_data
    
    def _draw_hand_landmarks(
        self,
        frame: np.ndarray,
        hand_landmarks
    ):
        """Dibuja landmarks de la mano"""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style()
        )
    
    def _extract_landmarks(self, hand_landmarks) -> List[dict]:
        """Extrae landmarks como lista de diccionarios"""
        landmarks = []
        
        for idx, landmark in enumerate(hand_landmarks.landmark):
            landmarks.append({
                'id': idx,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            })
        
        return landmarks
    
    def get_landmarks_array(self, hand_landmarks) -> np.ndarray:
        """Obtiene landmarks como array numpy"""
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def calculate_bounding_box(
        self,
        hand_landmarks,
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calcula bounding box de la mano
        
        Args:
            hand_landmarks: Landmarks de MediaPipe
            image_shape: (height, width) de la imagen
            
        Returns:
            Tuple de (x_min, y_min, x_max, y_max)
        """
        h, w = image_shape[:2]
        
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min = int(max(0, min(x_coords) - 20))
        x_max = int(min(w, max(x_coords) + 20))
        y_min = int(max(0, min(y_coords) - 20))
        y_max = int(min(h, max(y_coords) + 20))
        
        return x_min, y_min, x_max, y_max
    
    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Dibuja bounding box en el frame"""
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
        return frame
    
    def get_hand_orientation(self, landmarks: List[dict]) -> str:
        """
        Determina orientación de la mano
        
        Args:
            landmarks: Lista de landmarks
            
        Returns:
            'palm' o 'back'
        """
        # Usar landmarks 0 (muñeca) y 9 (medio)
        wrist_z = landmarks[0]['z']
        middle_z = landmarks[9]['z']
        
        return 'palm' if middle_z < wrist_z else 'back'
    
    def calculate_hand_distance(
        self,
        landmarks: List[dict],
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Calcula distancia estimada de la mano a la cámara
        
        Args:
            landmarks: Lista de landmarks
            image_shape: (height, width)
            
        Returns:
            Distancia estimada (valor relativo)
        """
        h, w = image_shape[:2]
        
        # Usar distancia entre muñeca y dedo medio
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        dx = (middle_tip['x'] - wrist['x']) * w
        dy = (middle_tip['y'] - wrist['y']) * h
        
        distance_2d = np.sqrt(dx**2 + dy**2)
        
        return distance_2d
    
    def close(self):
        """Cierra el rastreador"""
        if hasattr(self, 'hands'):
            self.hands.close()
    
    def __del__(self):
        """Limpieza"""
        self.close()