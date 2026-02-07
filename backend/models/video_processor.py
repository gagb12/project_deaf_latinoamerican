"""
Procesador de video para detección de señas
"""

import cv2
import numpy as np
from typing import Optional, Callable
import logging
import time
from threading import Thread, Lock
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Procesa video frame por frame para detección de señas
    """
    
    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        """
        Inicializa el procesador de video
        
        Args:
            source: Índice de la cámara o ruta de video
            width: Ancho del frame
            height: Alto del frame
            fps: Frames por segundo
        """
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        
        self.cap = None
        self.is_running = False
        self.lock = Lock()
        
        # Cola de frames
        self.frame_queue = Queue(maxsize=10)
        
        # Thread de captura
        self.capture_thread = None
        
        # Callback para procesamiento
        self.process_callback: Optional[Callable] = None
        
        # Métricas
        self.fps_actual = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
    
    def start(self) -> bool:
        """
        Inicia la captura de video
        
        Returns:
            True si se inició correctamente
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error("No se pudo abrir la cámara")
                return False
            
            # Configurar cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            self.is_running = True
            
            # Iniciar thread de captura
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"✅ Video iniciado: {self.width}x{self.height} @ {self.target_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando video: {e}")
            return False
    
    def stop(self):
        """Detiene la captura de video"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        # Limpiar cola
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info("Video detenido")
    
    def _capture_loop(self):
        """Loop de captura en thread separado"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("No se pudo leer frame")
                    continue
                
                # Actualizar FPS
                self._update_fps()
                
                # Agregar a cola si no está llena
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Delay para FPS objetivo
                time.sleep(1.0 / self.target_fps)
                
            except Exception as e:
                logger.error(f"Error en loop de captura: {e}")
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Obtiene el siguiente frame
        
        Args:
            timeout: Tiempo máximo de espera
            
        Returns:
            Frame o None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def process_frame(
        self,
        frame: np.ndarray,
        callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Procesa un frame con el callback registrado
        
        Args:
            frame: Frame a procesar
            callback: Función de procesamiento (opcional)
            
        Returns:
            Frame procesado
        """
        if callback:
            return callback(frame)
        elif self.process_callback:
            return self.process_callback(frame)
        else:
            return frame
    
    def set_process_callback(self, callback: Callable):
        """Establece callback de procesamiento"""
        self.process_callback = callback
    
    def _update_fps(self):
        """Actualiza cálculo de FPS"""
        self.frame_count += 1
        
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        
        if elapsed >= 1.0:
            self.fps_actual = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def get_fps(self) -> float:
        """Retorna FPS actual"""
        return self.fps_actual
    
    def resize_frame(
        self,
        frame: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        """
        Redimensiona un frame
        
        Args:
            frame: Frame a redimensionar
            width: Nuevo ancho (opcional)
            height: Nuevo alto (opcional)
            
        Returns:
            Frame redimensionado
        """
        if width is None and height is None:
            return frame
        
        h, w = frame.shape[:2]
        
        if width is None:
            width = int(w * height / h)
        elif height is None:
            height = int(h * width / w)
        
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    def flip_frame(
        self,
        frame: np.ndarray,
        horizontal: bool = True
    ) -> np.ndarray:
        """
        Voltea un frame
        
        Args:
            frame: Frame a voltear
            horizontal: Si True, voltea horizontalmente
            
        Returns:
            Frame volteado
        """
        return cv2.flip(frame, 1 if horizontal else 0)
    
    def add_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple = (10, 30),
        color: tuple = (0, 255, 0),
        font_scale: float = 1.0,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Agrega texto a un frame
        
        Args:
            frame: Frame
            text: Texto a agregar
            position: Posición (x, y)
            color: Color BGR
            font_scale: Escala de fuente
            thickness: Grosor
            
        Returns:
            Frame con texto
        """
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        return frame
    
    def add_fps_counter(
        self,
        frame: np.ndarray,
        position: tuple = (10, 30)
    ) -> np.ndarray:
        """Agrega contador de FPS al frame"""
        fps_text = f"FPS: {self.fps_actual:.1f}"
        return self.add_text(frame, fps_text, position)
    
    def get_info(self) -> dict:
        """Retorna información del video"""
        if self.cap:
            return {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps_target': self.target_fps,
                'fps_actual': self.fps_actual,
                'is_running': self.is_running,
                'queue_size': self.frame_queue.qsize()
            }
        return {}
    
    def __del__(self):
        """Limpieza"""
        self.stop()