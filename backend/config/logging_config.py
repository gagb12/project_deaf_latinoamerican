"""
Configuración de logging
"""

import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime


def setup_logging(app):
    """Configurar sistema de logging"""
    
    # Crear directorio de logs si no existe
    log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Nivel de logging
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    
    # Formato de logs
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para archivo general
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=app.config.get('LOG_MAX_BYTES', 10485760),
        backupCount=app.config.get('LOG_BACKUP_COUNT', 10)
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Handler para errores
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'errors.log'),
        maxBytes=10485760,
        backupCount=10
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    console_handler.setLevel(log_level)
    
    # Configurar logger de la app
    app.logger.addHandler(file_handler)
    app.logger.addHandler(error_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(log_level)
    
    # Deshabilitar logs de werkzeug en producción
    if not app.config.get('DEBUG'):
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
    
    app.logger.info('=' * 50)
    app.logger.info('Aplicación iniciada')
    app.logger.info(f'Ambiente: {app.config.get("ENV", "development")}')
    app.logger.info(f'Debug: {app.config.get("DEBUG")}')
    app.logger.info('=' * 50)


class RequestLogger:
    """Middleware para logging de requests"""
    
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        # Log de cada request
        path = environ.get('PATH_INFO', '')
        method = environ.get('REQUEST_METHOD', '')
        
        self.app.logger.info(f"{method} {path}")
        
        return self.app(environ, start_response)