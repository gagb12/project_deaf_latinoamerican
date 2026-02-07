"""
Configuración principal de la aplicación
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Cargar variables de entorno
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '../../.env'))


class Config:
    """Configuración base"""
    
    # Información de la aplicación
    APP_NAME = 'Traductor Lengua de Señas'
    APP_VERSION = '1.0.0'
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Servidor
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    CORS_ALLOW_HEADERS = ['Content-Type', 'Authorization']
    
    # WebSocket
    SOCKETIO_MESSAGE_QUEUE = os.getenv('SOCKETIO_MESSAGE_QUEUE', None)
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_CORS_ALLOWED_ORIGINS = '*'
    
    # Modelos ML
    MODEL_DIR = os.path.join(basedir, '../models/trained_models')
    LSC_MODEL_PATH = os.path.join(MODEL_DIR, 'lsc_model.h5')
    ASL_MODEL_PATH = os.path.join(MODEL_DIR, 'asl_model.h5')
    
    # Configuración de video
    VIDEO_FRAME_WIDTH = 640
    VIDEO_FRAME_HEIGHT = 480
    VIDEO_FPS = 30
    DETECTION_CONFIDENCE = 0.7
    TRACKING_CONFIDENCE = 0.5
    
    # Procesamiento de frames
    FRAME_SKIP = 2  # Procesar 1 de cada N frames
    MAX_BUFFER_SIZE = 30  # Máximo de frames en buffer
    
    # Cache
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Rate Limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = '100 per hour'
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'memory://')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(basedir, '../../logs/app.log')
    LOG_MAX_BYTES = 10485760  # 10MB
    LOG_BACKUP_COUNT = 10
    
    # Base de datos (opcional)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(basedir, '../../data/app.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Sesiones
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Seguridad
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Uploads (si se permiten)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    UPLOAD_FOLDER = os.path.join(basedir, '../../uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'webm'}
    
    # API Keys (si se usan servicios externos)
    GOOGLE_TRANSLATE_API_KEY = os.getenv('GOOGLE_TRANSLATE_API_KEY', '')
    
    # Mediapipe
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # TensorFlow
    TF_CPP_MIN_LOG_LEVEL = '2'  # Reducir logs de TensorFlow
    
    @staticmethod
    def init_app(app):
        """Inicialización adicional de la app"""
        pass


class DevelopmentConfig(Config):
    """Configuración de desarrollo"""
    DEBUG = True
    TESTING = False
    
    # Logging más detallado
    LOG_LEVEL = 'DEBUG'
    
    # CORS más permisivo
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']
    
    # Sin HTTPS en desarrollo
    SESSION_COOKIE_SECURE = False
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        print('🔧 Modo DESARROLLO activado')


class ProductionConfig(Config):
    """Configuración de producción"""
    DEBUG = False
    TESTING = False
    
    # Seguridad estricta
    SECRET_KEY = os.getenv('SECRET_KEY')
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # CORS específico
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '').split(',')
    
    # Logging a archivo
    LOG_LEVEL = 'WARNING'
    
    # Base de datos productiva
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    
    # Rate limiting estricto
    RATELIMIT_DEFAULT = '50 per hour'
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        
        # Enviar errores críticos por email
        import logging
        from logging.handlers import SMTPHandler
        
        credentials = None
        secure = None
        
        if os.getenv('MAIL_USERNAME'):
            credentials = (os.getenv('MAIL_USERNAME'), os.getenv('MAIL_PASSWORD'))
            if os.getenv('MAIL_USE_TLS'):
                secure = ()
        
        mail_handler = SMTPHandler(
            mailhost=(os.getenv('MAIL_SERVER', 'localhost'), 
                     os.getenv('MAIL_PORT', 25)),
            fromaddr=os.getenv('MAIL_DEFAULT_SENDER', 'noreply@example.com'),
            toaddrs=[os.getenv('ADMIN_EMAIL', 'admin@example.com')],
            subject='Error en Aplicación de Señas',
            credentials=credentials,
            secure=secure
        )
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)
        
        print('🚀 Modo PRODUCCIÓN activado')


class TestingConfig(Config):
    """Configuración de testing"""
    TESTING = True
    DEBUG = True
    
    # Base de datos en memoria
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Sin rate limiting en tests
    RATELIMIT_ENABLED = False
    
    # Logs mínimos
    LOG_LEVEL = 'ERROR'
    
    # WebSocket deshabilitado en tests
    SOCKETIO_ASYNC_MODE = 'threading'
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        print('🧪 Modo TESTING activado')