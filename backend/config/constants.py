"""
Constantes de la aplicación
"""


class Constants:
    """Constantes generales"""
    
    # Idiomas soportados
    SUPPORTED_LANGUAGES = ['LSC', 'ASL']
    
    # Estados de predicción
    PREDICTION_STATES = {
        'IDLE': 'idle',
        'DETECTING': 'detecting',
        'PREDICTING': 'predicting',
        'SUCCESS': 'success',
        'ERROR': 'error'
    }
    
    # Códigos de error
    ERROR_CODES = {
        'NO_HANDS_DETECTED': 1001,
        'LOW_CONFIDENCE': 1002,
        'MODEL_NOT_LOADED': 1003,
        'INVALID_LANGUAGE': 1004,
        'PROCESSING_ERROR': 1005,
        'CAMERA_ERROR': 1006
    }
    
    # Mensajes de error
    ERROR_MESSAGES = {
        1001: 'No se detectaron manos en el frame',
        1002: 'Confianza de predicción muy baja',
        1003: 'Modelo no cargado correctamente',
        1004: 'Idioma no soportado',
        1005: 'Error procesando frame',
        1006: 'Error accediendo a la cámara'
    }
    
    # Configuración de video
    VIDEO_CONFIG = {
        'WIDTH': 640,
        'HEIGHT': 480,
        'FPS': 30,
        'FORMAT': 'MJPEG'
    }
    
    # Límites
    MAX_FRAME_SIZE = 1024 * 1024  # 1MB
    MAX_PREDICTIONS_PER_SECOND = 10
    MAX_CONCURRENT_USERS = 100
    
    # Tiempos (en segundos)
    PREDICTION_TIMEOUT = 5
    CONNECTION_TIMEOUT = 30
    CACHE_TTL = 300  # 5 minutos
    
    # Palabras comunes LSC
    LSC_COMMON_WORDS = [
        'Hola', 'Gracias', 'Por favor', 'Sí', 'No',
        'Ayuda', 'Perdón', 'Entiendo', 'No entiendo',
        'Buenos días', 'Buenas tardes', 'Buenas noches'
    ]
    
    # Alfabeto ASL
    ASL_ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')