"""
detector.py - Detección de manos, pose y cara con MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging
from .config import config, MediaPipeConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Resultado de detección de un frame"""
    # Landmarks de manos (21 puntos × 3 coords cada una)
    left_hand: Optional[np.ndarray] = None
    right_hand: Optional[np.ndarray] = None

    # Landmarks de pose (33 puntos × 4 valores)
    pose: Optional[np.ndarray] = None

    # Landmarks faciales (468 puntos × 3 coords)
    face: Optional[np.ndarray] = None

    # Handedness (cuál mano es cuál)
    handedness: Optional[List[str]] = None

    # Frame procesado con dibujos
    annotated_frame: Optional[np.ndarray] = None

    # Metadatos
    hands_detected: int = 0
    pose_detected: bool = False
    face_detected: bool = False
    timestamp: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """
        Convierte todos los landmarks a un vector de features unificado

        Returns:
            Vector numpy con todas las features concatenadas
        """
        features = []

        # Mano izquierda (63 features) o ceros
        if self.left_hand is not None:
            features.extend(self.left_hand.flatten())
        else:
            features.extend([0.0] * 63)

        # Mano derecha (63 features) o ceros
        if self.right_hand is not None:
            features.extend(self.right_hand.flatten())
        else:
            features.extend([0.0] * 63)

        # Pose (132 features) o ceros
        if self.pose is not None:
            features.extend(self.pose.flatten())
        else:
            features.extend([0.0] * 132)

        return np.array(features, dtype=np.float32)

    @property
    def has_any_detection(self) -> bool:
        return self.hands_detected > 0 or self.pose_detected


class BodyDetector:
    """
    Detector de cuerpo completo para reconocimiento de señas.
    Usa MediaPipe para detectar manos, pose y expresiones faciales.
    """

    def __init__(self, mp_config: Optional[MediaPipeConfig] = None):
        self.config = mp_config or config.mediapipe

        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

        # Usar Holistic para mejor rendimiento (todo junto)
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            model_complexity=self.config.model_complexity,
        )

        # Alternativamente, componentes separados para más control
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

        if self.config.enable_pose:
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                model_complexity=self.config.model_complexity,
            )

        if self.config.enable_face_mesh:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=self.config.face_mesh_max_faces,
                refine_landmarks=self.config.face_mesh_refine_landmarks,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )

        logger.info("BodyDetector inicializado correctamente")

    def detect_holistic(self, frame: np.ndarray,
                         draw: bool = False) -> DetectionResult:
        """
        Detecta manos, pose y cara usando MediaPipe Holistic

        Args:
            frame: Frame BGR de OpenCV
            draw: Si dibujar landmarks en el frame

        Returns:
            DetectionResult con todos los landmarks
        """
        result = DetectionResult()

        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Procesar
        holistic_results = self.holistic.process(frame_rgb)

        frame_rgb.flags.writeable = True

        # Extraer mano izquierda
        if holistic_results.left_hand_landmarks:
            result.left_hand = self._landmarks_to_array(
                holistic_results.left_hand_landmarks, 21, 3
            )
            result.hands_detected += 1

        # Extraer mano derecha
        if holistic_results.right_hand_landmarks:
            result.right_hand = self._landmarks_to_array(
                holistic_results.right_hand_landmarks, 21, 3
            )
            result.hands_detected += 1

        # Extraer pose
        if holistic_results.pose_landmarks:
            result.pose = self._pose_landmarks_to_array(
                holistic_results.pose_landmarks
            )
            result.pose_detected = True

        # Extraer cara
        if holistic_results.face_landmarks:
            result.face = self._landmarks_to_array(
                holistic_results.face_landmarks, 468, 3
            )
            result.face_detected = True

        # Dibujar si se solicita
        if draw:
            annotated = frame.copy()
            self._draw_landmarks(annotated, holistic_results)
            result.annotated_frame = annotated

        return result

    def detect_hands_only(self, frame: np.ndarray) -> DetectionResult:
        """
        Detecta solo manos (más rápido)
        """
        result = DetectionResult()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)

        if hand_results.multi_hand_landmarks:
            handedness_list = []
            if hand_results.multi_handedness:
                for h in hand_results.multi_handedness:
                    handedness_list.append(
                        h.classification[0].label
                    )

            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                landmarks = self._landmarks_to_array(hand_landmarks, 21, 3)

                # Determinar si es izquierda o derecha
                if i < len(handedness_list):
                    # MediaPipe reporta "Left" y "Right" (espejado)
                    if handedness_list[i] == "Left":
                        result.right_hand = landmarks  # Espejado
                    else:
                        result.left_hand = landmarks
                else:
                    if result.right_hand is None:
                        result.right_hand = landmarks
                    else:
                        result.left_hand = landmarks

                result.hands_detected += 1

            result.handedness = handedness_list

        return result

    def _landmarks_to_array(
        self,
        landmarks,
        num_points: int,
        coords: int = 3
    ) -> np.ndarray:
        """Convierte landmarks de MediaPipe a numpy array"""
        points = []
        for lm in landmarks.landmark[:num_points]:
            points.extend([lm.x, lm.y, lm.z][:coords])
        return np.array(points, dtype=np.float32)

    def _pose_landmarks_to_array(self, landmarks) -> np.ndarray:
        """Convierte pose landmarks (incluye visibility)"""
        points = []
        for lm in landmarks.landmark:
            points.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(points, dtype=np.float32)

    def _draw_landmarks(self, frame: np.ndarray, results) -> None:
        """Dibuja todos los landmarks en el frame"""
        # Manos
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        # Pose (solo parte superior del cuerpo)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(80, 110, 255),
                    thickness=2,
                    circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(80, 256, 121),
                    thickness=2
                )
            )

    def close(self):
        """Liberar recursos"""
        self.holistic.close()
        self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("BodyDetector cerrado")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()