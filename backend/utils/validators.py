"""
Validadores para datos de entrada
"""

import re
import base64
import numpy as np
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def validate_language(language: str) -> bool:
    """
    Valida que el idioma sea soportado
    
    Args:
        language: Código del idioma
        
    Returns:
        True si es válido
    """
    supported_languages = ['LSC', 'ASL']
    return language.upper() in supported_languages


def validate_confidence(confidence: float) -> bool:
    """
    Valida que la confianza esté en rango válido
    
    Args:
        confidence: Valor de confianza
        
    Returns:
        True si está entre 0 y 1
    """
    try:
        conf = float(confidence)
        return 0.0 <= conf <= 1.0
    except (ValueError, TypeError):
        return False


def validate_base64_image(base64_string: str) -> bool:
    """
    Valida que un string sea una imagen base64 válida
    
    Args:
        base64_string: String en base64
        
    Returns:
        True si es válido
    """
    try:
        # Remover prefijo data:image si existe
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Intentar decodificar
        decoded = base64.b64decode(base64_string)
        
        # Verificar que tenga contenido
        if len(decoded) == 0:
            return False
        
        # Verificar magic numbers de imágenes comunes
        # JPEG: FF D8 FF
        # PNG: 89 50 4E 47
        # GIF: 47 49 46
        magic_numbers = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89\x50\x4e\x47',  # PNG
            b'\x47\x49\x46'  # GIF
        ]
        
        return any(decoded.startswith(magic) for magic in magic_numbers)
        
    except Exception as e:
        logger.debug(f"Error validando base64: {e}")
        return False


def validate_email(email: str) -> bool:
    """
    Valida formato de email
    
    Args:
        email: Email a validar
        
    Returns:
        True si es válido
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_username(username: str) -> bool:
    """
    Valida username
    
    Args:
        username: Username a validar
        
    Returns:
        True si es válido
    """
    # 3-20 caracteres, solo alfanuméricos y guión bajo
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None


def validate_session_id(session_id: str) -> bool:
    """
    Valida ID de sesión
    
    Args:
        session_id: ID a validar
        
    Returns:
        True si es válido
    """
    # Alfanumérico, guiones y guiones bajos, 10-100 caracteres
    pattern = r'^[a-zA-Z0-9_-]{10,100}$'
    return re.match(pattern, session_id) is not None


def validate_frame_dimensions(
    width: int,
    height: int,
    min_width: int = 160,
    max_width: int = 1920,
    min_height: int = 120,
    max_height: int = 1080
) -> bool:
    """
    Valida dimensiones de frame
    
    Args:
        width: Ancho
        height: Alto
        min_width: Ancho mínimo
        max_width: Ancho máximo
        min_height: Alto mínimo
        max_height: Alto máximo
        
    Returns:
        True si son válidas
    """
    return (
        min_width <= width <= max_width and
        min_height <= height <= max_height
    )


def validate_landmarks(landmarks: Any) -> bool:
    """
    Valida array de landmarks
    
    Args:
        landmarks: Array de landmarks
        
    Returns:
        True si es válido
    """
    try:
        if not isinstance(landmarks, (list, np.ndarray)):
            return False
        
        # Convertir a numpy array
        landmarks_array = np.array(landmarks)
        
        # Debe tener 63 valores (21 puntos * 3 coordenadas)
        if landmarks_array.shape != (63,):
            return False
        
        # Valores deben estar en rango razonable
        if not np.all((landmarks_array >= -2) & (landmarks_array <= 2)):
            return False
        
        return True
        
    except Exception as e:
        logger.debug(f"Error validando landmarks: {e}")
        return False


def validate_prediction_result(prediction: dict) -> bool:
    """
    Valida estructura de resultado de predicción
    
    Args:
        prediction: Diccionario de predicción
        
    Returns:
        True si es válido
    """
    required_fields = ['text', 'confidence', 'class_id', 'language']
    
    if not isinstance(prediction, dict):
        return False
    
    # Verificar campos requeridos
    if not all(field in prediction for field in required_fields):
        return False
    
    # Validar tipos
    if 