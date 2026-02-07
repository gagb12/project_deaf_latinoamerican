"""
Configuración de base de datos
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DatabaseConfig:
    """Configuración de base de datos"""
    
    def __init__(self, app=None):
        self.engine = None
        self.session = None
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Inicializar base de datos con Flask app"""
        database_uri = app.config.get('SQLALCHEMY_DATABASE_URI')
        
        # Crear engine
        self.engine = create_engine(
            database_uri,
            echo=app.config.get('DEBUG', False),
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Crear sesión
        self.session = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        )
        
        Base.query = self.session.query_property()
        
        # Crear tablas
        Base.metadata.create_all(bind=self.engine)
        
        # Cleanup al cerrar app
        @app.teardown_appcontext
        def shutdown_session(exception=None):
            self.session.remove()
    
    def create_tables(self):
        """Crear todas las tablas"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Eliminar todas las tablas"""
        Base.metadata.drop_all(bind=self.engine)


# Modelos de ejemplo (opcional)
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Text


class Translation(Base):
    """Modelo para guardar traducciones"""
    __tablename__ = 'translations'
    
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    language = Column(String(10), nullable=False)  # LSC o ASL
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(100))
    
    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'language': self.language,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id
        }


class User(Base):
    """Modelo de usuario (opcional)"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }