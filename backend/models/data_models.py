"""
Modelos de datos para la base de datos
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Translation(Base):
    """Modelo para traducciones guardadas"""
    __tablename__ = 'translations'
    
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    language = Column(String(10), nullable=False)  # LSC o ASL
    confidence = Column(Float, nullable=False)
    class_id = Column(Integer)
    session_id = Column(String(100))
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relación con usuario
    user = relationship("User", back_populates="translations")
    
    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'language': self.language,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __repr__(self):
        return f"<Translation(id={self.id}, text='{self.text}', language='{self.language}')>"


class User(Base):
    """Modelo de usuario (opcional)"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relación con traducciones
    translations = relationship("Translation", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Session(Base):
    """Modelo de sesión de uso"""
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    language = Column(String(10))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    total_predictions = Column(Integer, default=0)
    successful_predictions = Column(Integer, default=0)
    average_confidence = Column(Float, default=0.0)
    
    # Relación con usuario
    user = relationship("User", back_populates="sessions")
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'language': self.language,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'average_confidence': self.average_confidence
        }
    
    @property
    def duration(self):
        """Duración de la sesión en segundos"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    @property
    def success_rate(self):
        """Tasa de éxito de predicciones"""
        if self.total_predictions > 0:
            return self.successful_predictions / self.total_predictions
        return 0.0
    
    def __repr__(self):
        return f"<Session(id={self.id}, session_id='{self.session_id}')>"


class ModelMetrics(Base):
    """Métricas de los modelos"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    language = Column(String(10), nullable=False)
    version = Column(String(20))
    accuracy = Column(Float)
    loss = Column(Float)
    num_classes = Column(Integer)
    training_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    notes = Column(Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'language': self.language,
            'version': self.version,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'num_classes': self.num_classes,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'is_active': self.is_active,
            'notes': self.notes
        }
    
    def __repr__(self):
        return f"<ModelMetrics(language='{self.language}', accuracy={self.accuracy})>"