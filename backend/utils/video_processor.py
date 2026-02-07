"""
Utilidades para procesamiento de video
"""

import cv2
import numpy as np
import base64
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def process_video_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    flip: bool = True
) -> np.ndarray:
    """
    Procesa un frame de video
    
    Args:
        frame: Frame original
        target_size: Tamaño objetivo (width, height)
        flip: Si True, voltea horizontalmente
        
    Returns:
        Frame procesado
    """
    if flip:
        frame = cv2.flip(frame, 1)
    
    if target_size:
        frame = cv2.resize(frame, target_size)
    
    return frame


def encode_frame_to_base64(
    frame: np.ndarray,
    quality: int = 80,
    format: str = '.jpg'
) -> str:
    """
    Codifica un frame a base64
    
    Args:
        frame: Frame a codificar
        quality: Calidad de compresión (1-100)
        format: Formato de imagen ('.jpg' o '.png')
        
    Returns:
        String base64
    """
    try:
        # Codificar a formato de imagen
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(format, frame, encode_params)
        
        if not success:
            raise ValueError("Error codificando imagen")
        
        # Convertir a base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
        
    except Exception as e:
        logger.error(f"Error codificando frame: {e}")
        raise


def decode_base64_to_frame(
    base64_string: str,
    remove_prefix: bool = True
) -> np.ndarray:
    """
    Decodifica base64 a frame
    
    Args:
        base64_string: String base64
        remove_prefix: Si True, remueve prefijo data:image
        
    Returns:
        Frame decodificado
    """
    try:
        # Remover prefijo si existe
        if remove_prefix and ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decodificar base64
        img_data = base64.b64decode(base64_string)
        
        # Convertir a numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decodificar imagen
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Error decodificando imagen")
        
        return frame
        
    except Exception as e:
        logger.error(f"Error decodificando base64: {e}")
        raise


def apply_filters(
    frame: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0
) -> np.ndarray:
    """
    Aplica filtros a un frame
    
    Args:
        frame: Frame original
        brightness: Factor de brillo (1.0 = sin cambio)
        contrast: Factor de contraste (1.0 = sin cambio)
        saturation: Factor de saturación (1.0 = sin cambio)
        
    Returns:
        Frame con filtros aplicados
    """
    # Ajustar brillo y contraste
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness * 50 - 50)
    
    # Ajustar saturación
    if saturation != 1.0:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return adjusted


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.0,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 10
) -> np.ndarray:
    """
    Dibuja texto con fondo en un frame
    
    Args:
        frame: Frame donde dibujar
        text: Texto a dibujar
        position: Posición (x, y)
        font_scale: Escala de fuente
        thickness: Grosor del texto
        text_color: Color del texto (BGR)
        bg_color: Color del fondo (BGR)
        padding: Padding alrededor del texto
        
    Returns:
        Frame con texto dibujado
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Obtener tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Dibujar fondo
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Dibujar texto
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
    
    return frame


def create_mosaic(
    frames: list,
    grid_size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Crea un mosaico de múltiples frames
    
    Args:
        frames: Lista de frames
        grid_size: Tamaño de la grilla (rows, cols)
        
    Returns:
        Frame mosaico
    """
    if not frames:
        raise ValueError("Lista de frames vacía")
    
    num_frames = len(frames)
    
    if grid_size is None:
        # Calcular grilla automáticamente
        cols = int(np.ceil(np.sqrt(num_frames)))
        rows = int(np.ceil(num_frames / cols))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # Redimensionar todos los frames al mismo tamaño
    target_h = frames[0].shape[0] // rows
    target_w = frames[0].shape[1] // cols
    
    resized_frames = [
        cv2.resize(frame, (target_w, target_h))
        for frame in frames
    ]
    
    # Rellenar con frames negros si es necesario
    while len(resized_frames) < rows * cols:
        resized_frames.append(np.zeros_like(resized_frames[0]))
    
    # Crear mosaico
    mosaic_rows = []
    for i in range(rows):
        row_frames = resized_frames[i * cols:(i + 1) * cols]
        mosaic_rows.append(np.hstack(row_frames))
    
    mosaic = np.vstack(mosaic_rows)
    
    return mosaic


def extract_roi(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 0
) -> np.ndarray:
    """
    Extrae región de interés del frame
    
    Args:
        frame: Frame original
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        padding: Padding adicional
        
    Returns:
        ROI extraída
    """
    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    
    # Aplicar padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    roi = frame[y_min:y_max, x_min:x_max]
    
    return roi