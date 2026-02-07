"""
Paquete de configuración del backend
Importa y expone todas las configuraciones
"""

from .settings import Config, DevelopmentConfig, ProductionConfig, TestingConfig
from .database import DatabaseConfig
from .ml_config import MLConfig
from .cors_config import CORSConfig
from .websocket_config import WebSocketConfig
from .logging_config import setup_logging

__all__ = [
    'Config',
    'DevelopmentConfig',
    'ProductionConfig',
    'TestingConfig',
    'DatabaseConfig',
    'MLConfig',
    'CORSConfig',
    'WebSocketConfig',
    'setup_logging'
]

# Obtener configuración según ambiente
import os

config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Retorna la configuración según el ambiente"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config_by_name.get(env, DevelopmentConfig)