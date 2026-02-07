"""
predictor.py - Predicción en tiempo real de señas
"""

import torch
import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .model import create_model
from .detector import BodyDetector
from .feature_extractor import FeatureExtractor
from .config import config, ModelConfig
import logging
import time

logger = logging.getLogger(__name__)


class SignPredictor:
    """
    Predictor en tiempo real de lengua de señas.
    Procesa video de la cámara y predice señas continuamente.
    """

    def __init__(
        self,
        model_path: str,
        language: str = "LSC",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
        cooldown_frames: int = 15
    ):
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Cargar modelo
        self.model, self.classes, self.class_to_idx = self._load_model(model_path)
        self.model.eval()

        # Componentes
        self.detector = BodyDetector()
        self.feature_extractor = FeatureExtractor()

        # Buffer de secuencia
        self.sequence_buffer = deque(maxlen=config.model.sequence_length)

        # Estado
        self.sentence: List[str] = []
        self.last_prediction = ""
        self.cooldown_counter = 0
        self.prediction_history = deque(maxlen=10)

        # Métricas de rendimiento
        self.fps_counter = deque(maxlen=30)

        logger.info(
            f"SignPredictor inicializado: {language}, "
            f"{len(self.classes)} clases, device={self.device}"
        )

    def _load_model(self, model_path: str):
        """Carga modelo entrenado"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model_cfg = checkpoint['model_config']
        model = create_model(
            ModelConfig(
                architecture=model_cfg['architecture'],
                input_features=model_cfg['input_features'],
                hidden_size=model_cfg['hidden_size'],
                num_layers=model_cfg['num_layers'],
                num_heads=model_cfg.get('num_heads', 8),
                dropout=model_cfg['dropout'],
            ),
            num_classes=model_cfg['num_classes']
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        classes = checkpoint['classes']
        class_to_idx = checkpoint['class_to_idx']

        logger.info(f"Modelo cargado: {model_path}")
        return model, classes, class_to_idx

    def predict_frame(self, frame: np.ndarray) -> Dict:
        """
        Procesa un frame y devuelve predicción

        Args:
            frame: Frame BGR de OpenCV

        Returns:
            Diccionario con predicción y metadatos
        """
        start_time = time.time()

        # Detectar landmarks
        detection = self.detector.detect_holistic(frame, draw=True)

        result = {
            'prediction': None,
            'confidence': 0.0,
            'sentence': ' '.join(self.sentence),
            'detection': detection,
            'fps': 0,
            'alternatives': []
        }

        if not detection.has_any_detection:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)
            return result

        # Extraer features
        features = self.feature_extractor.extract(detection)
        self.sequence_buffer.append(features)

        # Necesitamos suficientes frames
        if len(self.sequence_buffer) < config.model.sequence_length:
            return result

        # Predecir
        sequence = np.array(list(self.sequence_buffer), dtype=np.float32)
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=-1).cpu().numpy()[0]

        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])

        # Aplicar umbral y cooldown
        if confidence >= self.confidence_threshold:
            predicted_word = self.classes[predicted_idx]

            if self.cooldown_counter <= 0 and predicted_word != self.last_prediction:
                self.sentence.append(predicted_word)
                self.last_prediction = predicted_word
                self.cooldown_counter = self.cooldown_frames
                self.prediction_history.append({
                    'word': predicted_word,
                    'confidence': confidence,
                    'timestamp': time.time()
                })

            result['prediction'] = predicted_word
            result['confidence'] = confidence

        # Top-5 alternativas
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        result['alternatives'] = [
            {
                'word': self.classes[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top5_indices
        ]

        result['sentence'] = ' '.join(self.sentence)

        # Cooldown
        self.cooldown_counter = max(0, self.cooldown_counter - 1)

        # FPS
        elapsed = time.time() - start_time
        self.fps_counter.append(elapsed)
        result['fps'] = 1.0 / (sum(self.fps_counter) / len(self.fps_counter))

        return result

    def run_realtime(self, camera_id: int = 0):
        """
        Ejecuta predicción en tiempo real con interfaz visual

        Args:
            camera_id: ID de la cámara
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        logger.info("Iniciando predicción en tiempo real...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Predecir
            result = self.predict_frame(frame)

            # Dibujar interfaz
            display = self._draw_prediction_ui(frame, result)

            cv2.imshow(f'SignAI - {self.language}', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Limpiar oración
                self.sentence.clear()
                self.last_prediction = ""
            elif key == ord('r'):
                # Reset buffer
                self.sequence_buffer.clear()
                self.feature_extractor.reset()

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()

    def _draw_prediction_ui(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Dibuja la interfaz de predicción"""
        h, w = frame.shape[:2]
        display = result['detection'].annotated_frame if result['detection'].annotated_frame is not None else frame.copy()

        # Panel superior - Predicción actual
        cv2.rectangle(display, (0, 0), (w, 100), (20, 20, 60), -1)

        if result['prediction']:
            word = result['prediction'].replace('_', ' ').upper()
            conf = result['confidence']

            # Color según confianza
            if conf > 0.9:
                color = (0, 255, 0)
            elif conf > 0.7:
                color = (0, 200, 255)
            else:
                color = (0, 100, 255)

            cv2.putText(display, word, (20, 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
            cv2.putText(display, f"{conf:.0%}", (w - 120, 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Barra de confianza
            bar_w = int((w - 40) * conf)
            cv2.rectangle(display, (20, 70), (20 + bar_w, 85), color, -1)
            cv2.rectangle(display, (20, 70), (w - 20, 85), (100, 100, 100), 1)

        else:
            cv2.putText(display, "Esperando seña...", (20, 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

        # Panel inferior - Oración formada
        cv2.rectangle(display, (0, h - 80), (w, h), (20, 20, 60), -1)

        sentence_display = ' '.join(self.sentence[-8:])  # Últimas 8 palabras
        if sentence_display:
            cv2.putText(display, sentence_display, (20, h - 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(display, "Haz una seña para empezar...", (20, h - 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        # FPS
        cv2.putText(display, f"FPS: {result['fps']:.0f}", (w - 130, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Alternativas (panel lateral derecho)
        if result['alternatives']:
            panel_x = w - 250
            cv2.rectangle(display, (panel_x, 110), (w, 110 + 30 * 5), (30, 30, 70), -1)
            cv2.putText(display, "Top 5:", (panel_x + 10, 135),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            for i, alt in enumerate(result['alternatives'][:5]):
                y = 155 + i * 25
                word = alt['word'].replace('_', ' ')
                conf = alt['confidence']
                cv2.putText(display, f"{word}: {conf:.0%}",
                             (panel_x + 10, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                             (180, 180, 180), 1)

        # Controles
        cv2.putText(display, "C=Limpiar | R=Reset | Q=Salir",
                     (20, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        return display

    def reset(self):
        """Resetea todo el estado"""
        self.sequence_buffer.clear()
        self.sentence.clear()
        self.last_prediction = ""
        self.cooldown_counter = 0
        self.feature_extractor.reset()

    def close(self):
        """Libera recursos"""
        self.detector.close()