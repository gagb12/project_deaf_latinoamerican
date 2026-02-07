"""
data_collector.py - Herramienta para recolectar datos de señas
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from .detector import BodyDetector, DetectionResult
from .feature_extractor import FeatureExtractor
import logging

logger = logging.getLogger(__name__)


class SignDataCollector:
    """
    Herramienta interactiva para recolectar datos de señas
    con la cámara. Diseñada para ser usada por personas sordas
    con interfaz visual clara.
    """

    def __init__(
        self,
        language: str = "LSC",
        output_dir: str = "./data/raw",
        sequence_length: int = 30,
        fps: int = 30
    ):
        self.language = language
        self.output_dir = Path(output_dir) / language.lower()
        self.sequence_length = sequence_length
        self.fps = fps

        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Componentes
        self.detector = BodyDetector()
        self.feature_extractor = FeatureExtractor()

        # Estado
        self.current_word = ""
        self.recording = False
        self.current_sequence: List[np.ndarray] = []
        self.recorded_count: Dict[str, int] = {}

        # Cargar conteo existente
        self._load_existing_counts()

        # Vocabulario por lengua de señas
        self.vocabulary = self._load_vocabulary()

    def _load_vocabulary(self) -> List[str]:
        """Carga vocabulario de la lengua de señas seleccionada"""
        vocab_files = {
            "LSC": "lsc_vocabulary.json",
            "LSM": "lsm_vocabulary.json",
            "LSE": "lse_vocabulary.json",
            "ASL": "asl_vocabulary.json",
        }

        vocab_path = Path(f"./sign_recognition/languages/vocabulary/{vocab_files.get(self.language, 'lsc_vocabulary.json')}")

        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('words', [])

        # Vocabulario básico por defecto
        return [
            "hola", "gracias", "por_favor", "si", "no",
            "ayuda", "nombre", "como_estas", "bien", "mal",
            "agua", "comida", "familia", "amigo", "escuela",
            "casa", "trabajo", "doctor", "emergencia", "policia",
            "yo", "tu", "el", "ella", "nosotros",
            "querer", "poder", "necesitar", "entender", "saber",
            "donde", "cuando", "que", "como", "porque",
            "grande", "pequeño", "mucho", "poco", "rapido",
            "lento", "bonito", "feo", "nuevo", "viejo",
        ]

    def _load_existing_counts(self):
        """Carga conteo de muestras existentes"""
        for word_dir in self.output_dir.iterdir():
            if word_dir.is_dir():
                count = len(list(word_dir.glob("*.npy")))
                self.recorded_count[word_dir.name] = count

    def collect_interactive(self):
        """
        Modo interactivo de recolección con la cámara.
        Interfaz visual clara para personas sordas.
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        word_index = 0
        self.current_word = self.vocabulary[word_index]
        countdown = -1
        countdown_start = 0

        print(f"\n{'='*60}")
        print(f"  📹 SignAI - Recolector de Datos de {self.language}")
        print(f"  Vocabulario: {len(self.vocabulary)} palabras")
        print(f"{'='*60}")
        print("\nControles:")
        print("  ESPACIO → Iniciar grabación (3 seg countdown)")
        print("  N → Siguiente palabra")
        print("  P → Palabra anterior")
        print("  Q → Salir")
        print(f"{'='*60}\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Espejo para que sea más natural
            frame = cv2.flip(frame, 1)

            # Detectar
            detection = self.detector.detect_holistic(frame, draw=True)
            display_frame = detection.annotated_frame if detection.annotated_frame is not None else frame.copy()

            # Interfaz visual
            self._draw_ui(display_frame, detection, countdown)

            # Countdown
            if countdown > 0:
                elapsed = cv2.getTickCount() / cv2.getTickFrequency() - countdown_start
                countdown = 3 - int(elapsed)
                if countdown <= 0:
                    self.recording = True
                    self.current_sequence = []
                    countdown = -1

            # Grabando
            if self.recording:
                if detection.has_any_detection:
                    features = self.feature_extractor.extract(detection)
                    self.current_sequence.append(features)

                # Barra de progreso visual
                progress = len(self.current_sequence) / self.sequence_length
                self._draw_recording_bar(display_frame, progress)

                # Flash visual verde
                if len(self.current_sequence) < 5:
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0),
                                  (display_frame.shape[1], display_frame.shape[0]),
                                  (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.1, display_frame, 0.9, 0, display_frame)

                # Completado
                if len(self.current_sequence) >= self.sequence_length:
                    self._save_sequence()
                    self.recording = False
                    self.feature_extractor.reset()

                    # Flash visual azul de confirmación
                    for _ in range(3):
                        flash = display_frame.copy()
                        cv2.rectangle(flash, (0, 0),
                                      (flash.shape[1], flash.shape[0]),
                                      (255, 200, 0), -1)
                        cv2.addWeighted(flash, 0.2, display_frame, 0.8, 0, flash)
                        cv2.imshow('SignAI Collector', flash)
                        cv2.waitKey(100)

            cv2.imshow('SignAI Collector', display_frame)

            # Teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not self.recording:
                countdown = 3
                countdown_start = cv2.getTickCount() / cv2.getTickFrequency()
            elif key == ord('n'):
                word_index = (word_index + 1) % len(self.vocabulary)
                self.current_word = self.vocabulary[word_index]
                self.recording = False
                self.current_sequence = []
            elif key == ord('p'):
                word_index = (word_index - 1) % len(self.vocabulary)
                self.current_word = self.vocabulary[word_index]
                self.recording = False
                self.current_sequence = []

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()

        self._print_summary()

    def _draw_ui(self, frame, detection, countdown):
        """Dibuja interfaz visual en el frame"""
        h, w = frame.shape[:2]

        # Fondo superior
        cv2.rectangle(frame, (0, 0), (w, 120), (20, 20, 60), -1)

        # Palabra actual
        word_display = self.current_word.replace('_', ' ').upper()
        cv2.putText(frame, f"Sena: {word_display}",
                     (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                     1.5, (255, 255, 255), 3)

        # Contador de muestras
        count = self.recorded_count.get(self.current_word, 0)
        color = (0, 255, 0) if count >= 30 else (0, 200, 255) if count >= 10 else (0, 0, 255)
        cv2.putText(frame, f"Muestras: {count}/30",
                     (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                     0.8, color, 2)

        # Estado de detección
        status_y = 90
        if detection.hands_detected > 0:
            cv2.putText(frame, f"Manos: {detection.hands_detected}",
                         (w - 250, status_y), cv2.FONT_HERSHEY_SIMPLEX,
                         0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Manos: NO",
                         (w - 250, status_y), cv2.FONT_HERSHEY_SIMPLEX,
                         0.7, (0, 0, 255), 2)

        # Countdown
        if countdown > 0:
            cv2.putText(frame, str(countdown),
                         (w // 2 - 40, h // 2 + 30),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         5.0, (0, 200, 255), 10)

        # Estado de grabación
        if self.recording:
            cv2.circle(frame, (w - 40, 40), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 100, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Instrucciones en la parte inferior
        cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 60), -1)
        cv2.putText(frame, "ESPACIO=Grabar | N=Siguiente | P=Anterior | Q=Salir",
                     (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, (200, 200, 200), 1)

    def _draw_recording_bar(self, frame, progress):
        """Barra de progreso de grabación"""
        h, w = frame.shape[:2]
        bar_h = 20
        bar_y = 125
        bar_w = int(w * 0.8)
        bar_x = int(w * 0.1)

        # Fondo
        cv2.rectangle(frame, (bar_x, bar_y),
                       (bar_x + bar_w, bar_y + bar_h),
                       (50, 50, 50), -1)

        # Progreso
        fill_w = int(bar_w * min(progress, 1.0))
        color = (0, 255, 0) if progress < 1.0 else (255, 200, 0)
        cv2.rectangle(frame, (bar_x, bar_y),
                       (bar_x + fill_w, bar_y + bar_h),
                       color, -1)

        # Texto
        percent = int(progress * 100)
        cv2.putText(frame, f"{percent}%",
                     (bar_x + bar_w + 10, bar_y + 16),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _save_sequence(self):
        """Guarda la secuencia grabada"""
        word_dir = self.output_dir / self.current_word
        word_dir.mkdir(exist_ok=True)

        # Ajustar longitud
        sequence = self.current_sequence[:self.sequence_length]
        while len(sequence) < self.sequence_length:
            sequence.append(sequence[-1])

        # Guardar como numpy
        data = np.array(sequence, dtype=np.float32)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.current_word}_{timestamp}.npy"
        filepath = word_dir / filename

        np.save(filepath, data)

        # Actualizar conteo
        self.recorded_count[self.current_word] = \
            self.recorded_count.get(self.current_word, 0) + 1

        logger.info(
            f"Guardado: {filename} "
            f"(Total: {self.recorded_count[self.current_word]})"
        )

    def _print_summary(self):
        """Imprime resumen de recolección"""
        print(f"\n{'='*60}")
        print(f"  📊 Resumen de Recolección - {self.language}")
        print(f"{'='*60}")

        total = 0
        for word in sorted(self.recorded_count.keys()):
            count = self.recorded_count[word]
            total += count
            bar = '█' * (count // 2) + '░' * max(0, 15 - count // 2)
            status = '✅' if count >= 30 else '⚠️' if count >= 10 else '❌'
            print(f"  {status} {word:20s} {bar} {count:3d}/30")

        print(f"\n  Total muestras: {total}")
        print(f"  Palabras con suficientes datos: "
              f"{sum(1 for c in self.recorded_count.values() if c >= 30)}"
              f"/{len(self.vocabulary)}")
        print(f"{'='*60}\n")