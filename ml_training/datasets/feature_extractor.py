"""
feature_extractor.py - Extracción de features para reconocimiento de señas
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .detector import DetectionResult
import logging

logger = logging.getLogger(__name__)


@dataclass
class HandFeatures:
    """Features calculadas de una mano"""
    # Ángulos entre dedos
    finger_angles: np.ndarray           # 5 ángulos
    # Distancias entre puntas de dedos
    fingertip_distances: np.ndarray     # 10 distancias (combinaciones)
    # Cuáles dedos están extendidos
    fingers_extended: np.ndarray        # 5 booleanos
    # Posición relativa al cuerpo
    hand_position_relative: np.ndarray  # 3 coords relativas
    # Orientación de la palma
    palm_orientation: np.ndarray        # 3 valores (normal de la palma)
    # Velocidad (diferencia con frame anterior)
    velocity: Optional[np.ndarray] = None


class FeatureExtractor:
    """
    Extrae features de alto nivel de los landmarks detectados.
    Estas features son más discriminativas que los landmarks raw
    para el reconocimiento de señas.
    """

    # Índices de MediaPipe para cada dedo
    FINGER_TIPS = [4, 8, 12, 16, 20]       # Pulgar, Índice, Medio, Anular, Meñique
    FINGER_PIPS = [3, 6, 10, 14, 18]        # Articulaciones PIP
    FINGER_MCPS = [2, 5, 9, 13, 17]         # Articulaciones MCP
    FINGER_DIPS = [3, 7, 11, 15, 19]        # Articulaciones DIP
    WRIST = 0

    # Puntos clave de pose para señas
    POSE_SHOULDERS = [11, 12]    # Hombros
    POSE_ELBOWS = [13, 14]       # Codos
    POSE_WRISTS = [15, 16]       # Muñecas
    POSE_NOSE = 0
    POSE_LEFT_EYE = 2
    POSE_RIGHT_EYE = 5

    def __init__(self):
        self.previous_features: Optional[np.ndarray] = None
        self.feature_history: List[np.ndarray] = []

    def extract(
        self,
        detection: DetectionResult,
        normalize: bool = True,
        include_velocity: bool = True,
        include_angles: bool = True
    ) -> np.ndarray:
        """
        Extrae vector de features completo de una detección

        Args:
            detection: Resultado de detección
            normalize: Si normalizar las features
            include_velocity: Si incluir velocidad
            include_angles: Si incluir ángulos entre dedos

        Returns:
            Vector de features numpy
        """
        features = []

        # 1. Features básicas de landmarks (raw)
        basic_features = detection.to_feature_vector()
        features.extend(basic_features)

        # 2. Features de mano derecha (si existe)
        if detection.right_hand is not None:
            right_hand_points = detection.right_hand.reshape(21, 3)
            hand_feats = self._extract_hand_features(right_hand_points)

            if include_angles:
                features.extend(hand_feats.finger_angles)
            features.extend(hand_feats.fingertip_distances)
            features.extend(hand_feats.fingers_extended.astype(float))
        else:
            # Padding
            if include_angles:
                features.extend([0.0] * 5)   # ángulos
            features.extend([0.0] * 10)       # distancias
            features.extend([0.0] * 5)        # dedos extendidos

        # 3. Features de mano izquierda (si existe)
        if detection.left_hand is not None:
            left_hand_points = detection.left_hand.reshape(21, 3)
            hand_feats = self._extract_hand_features(left_hand_points)

            if include_angles:
                features.extend(hand_feats.finger_angles)
            features.extend(hand_feats.fingertip_distances)
            features.extend(hand_feats.fingers_extended.astype(float))
        else:
            if include_angles:
                features.extend([0.0] * 5)
            features.extend([0.0] * 10)
            features.extend([0.0] * 5)

        # 4. Relación entre manos (si ambas existen)
        if detection.left_hand is not None and detection.right_hand is not None:
            inter_hand = self._extract_inter_hand_features(
                detection.left_hand.reshape(21, 3),
                detection.right_hand.reshape(21, 3)
            )
            features.extend(inter_hand)
        else:
            features.extend([0.0] * 10)

        # 5. Posición de manos relativa al cuerpo
        if detection.pose is not None:
            pose_points = detection.pose.reshape(33, 4)[:, :3]  # Solo xyz

            if detection.right_hand is not None:
                right_relative = self._hand_body_relation(
                    detection.right_hand.reshape(21, 3), pose_points
                )
                features.extend(right_relative)
            else:
                features.extend([0.0] * 6)

            if detection.left_hand is not None:
                left_relative = self._hand_body_relation(
                    detection.left_hand.reshape(21, 3), pose_points
                )
                features.extend(left_relative)
            else:
                features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 12)

        # 6. Velocidad (temporal)
        feature_vector = np.array(features, dtype=np.float32)

        if include_velocity and self.previous_features is not None:
            min_len = min(len(feature_vector), len(self.previous_features))
            velocity = feature_vector[:min_len] - self.previous_features[:min_len]
            # Añadir magnitud de velocidad
            features.append(float(np.linalg.norm(velocity)))
        else:
            features.append(0.0)

        feature_vector = np.array(features, dtype=np.float32)

        # Normalizar
        if normalize:
            feature_vector = self._normalize(feature_vector)

        # Guardar para cálculo de velocidad
        self.previous_features = feature_vector.copy()

        return feature_vector

    def _extract_hand_features(self, points: np.ndarray) -> HandFeatures:
        """
        Extrae features de alto nivel de una mano

        Args:
            points: Array (21, 3) con los landmarks de la mano
        """
        # Ángulos entre dedos
        finger_angles = np.zeros(5)
        for i, (tip, pip, mcp) in enumerate(
            zip(self.FINGER_TIPS, self.FINGER_PIPS, self.FINGER_MCPS)
        ):
            v1 = points[tip] - points[pip]
            v2 = points[mcp] - points[pip]
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            )
            finger_angles[i] = np.arccos(np.clip(cos_angle, -1, 1))

        # Distancias entre puntas de dedos
        tips = points[self.FINGER_TIPS]
        distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                distances.append(np.linalg.norm(tips[i] - tips[j]))
        fingertip_distances = np.array(distances)

        # Dedos extendidos
        fingers_extended = self._detect_extended_fingers(points)

        # Posición relativa (centrada en muñeca)
        wrist = points[self.WRIST]
        hand_center = np.mean(points, axis=0)
        hand_position_relative = hand_center - wrist

        # Orientación de la palma (normal usando puntos clave)
        v1 = points[5] - points[0]   # Muñeca → base índice
        v2 = points[17] - points[0]  # Muñeca → base meñique
        palm_normal = np.cross(v1, v2)
        palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)

        return HandFeatures(
            finger_angles=finger_angles,
            fingertip_distances=fingertip_distances,
            fingers_extended=fingers_extended,
            hand_position_relative=hand_position_relative,
            palm_orientation=palm_normal,
        )

    def _detect_extended_fingers(self, points: np.ndarray) -> np.ndarray:
        """
        Detecta qué dedos están extendidos

        Args:
            points: Landmarks de mano (21, 3)

        Returns:
            Array de 5 booleanos [pulgar, índice, medio, anular, meñique]
        """
        extended = np.zeros(5, dtype=bool)

        # Pulgar: comparar con base (dirección especial)
        # El pulgar se extiende lateralmente
        if points[4][0] > points[3][0]:  # Mano derecha
            extended[0] = points[4][0] > points[2][0]
        else:
            extended[0] = points[4][0] < points[2][0]

        # Otros dedos: la punta está más arriba que el PIP
        for i, (tip, pip) in enumerate(
            zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:])
        ):
            # En MediaPipe, Y crece hacia abajo
            extended[i + 1] = points[tip][1] < points[pip][1]

        return extended

    def _extract_inter_hand_features(
        self,
        left: np.ndarray,
        right: np.ndarray
    ) -> List[float]:
        """
        Features de relación entre ambas manos

        Args:
            left: Landmarks mano izquierda (21, 3)
            right: Landmarks mano derecha (21, 3)

        Returns:
            Lista de features inter-mano
        """
        features = []

        # Distancia entre muñecas
        wrist_distance = np.linalg.norm(left[0] - right[0])
        features.append(wrist_distance)

        # Distancia entre centros de manos
        left_center = np.mean(left, axis=0)
        right_center = np.mean(right, axis=0)
        center_distance = np.linalg.norm(left_center - right_center)
        features.append(center_distance)

        # ¿Manos se tocan? (distancia mínima entre puntas)
        min_tip_distance = float('inf')
        for lt in self.FINGER_TIPS:
            for rt in self.FINGER_TIPS:
                d = np.linalg.norm(left[lt] - right[rt])
                min_tip_distance = min(min_tip_distance, d)
        features.append(min_tip_distance)

        # Simetría (qué tan simétricas son las posiciones)
        left_relative = left - left[0]   # Relativa a muñeca izq
        right_relative = right - right[0]
        symmetry = np.mean(np.abs(left_relative + right_relative))
        features.append(symmetry)

        # ¿Manos están cruzadas?
        features.append(float(left_center[0] > right_center[0]))

        # Diferencia de altura entre manos
        features.append(left_center[1] - right_center[1])

        # Distancias entre puntas específicas (4 más relevantes)
        key_pairs = [(8, 8), (4, 8), (8, 4), (12, 12)]
        for li, ri in key_pairs:
            features.append(np.linalg.norm(left[li] - right[ri]))

        return features

    def _hand_body_relation(
        self,
        hand: np.ndarray,
        pose: np.ndarray
    ) -> List[float]:
        """
        Relación de posición de mano respecto al cuerpo

        Args:
            hand: Landmarks de mano (21, 3)
            pose: Landmarks de pose (33, 3)

        Returns:
            Features de relación mano-cuerpo
        """
        features = []
        hand_center = np.mean(hand, axis=0)

        # Distancia a la cara (nariz)
        nose = pose[self.POSE_NOSE]
        features.append(np.linalg.norm(hand_center - nose))

        # Posición relativa a hombros
        shoulder_center = (
            pose[self.POSE_SHOULDERS[0]] + pose[self.POSE_SHOULDERS[1]]
        ) / 2
        relative_to_shoulders = hand_center - shoulder_center
        features.extend(relative_to_shoulders.tolist())

        # ¿Mano está a la altura de la cara?
        eye_level = (
            pose[self.POSE_LEFT_EYE][1] + pose[self.POSE_RIGHT_EYE][1]
        ) / 2
        features.append(hand_center[1] - eye_level)

        # ¿Mano está frente al cuerpo?
        features.append(hand_center[2] - shoulder_center[2])

        return features

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normaliza features al rango [-1, 1]"""
        # Evitar división por cero
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features

    def reset(self):
        """Resetear estado temporal"""
        self.previous_features = None
        self.feature_history.clear()