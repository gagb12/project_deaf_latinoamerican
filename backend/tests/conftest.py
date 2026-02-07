"""
Configuración de pytest y fixtures compartidas
"""

import pytest
import os
import sys
import numpy as np
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

# Agregar backend al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server import app, socketio
from models.sign_detector import SignLanguageDetector
from models.video_processor import VideoProcessor
from config.settings import TestingConfig


@pytest.fixture
def flask_app():
    """Fixture de la aplicación Flask"""
    app.config.from_object(TestingConfig)
    
    with app.app_context():
        yield app


@pytest.fixture
def client(flask_app):
    """Cliente de prueba de Flask"""
    return flask_app.test_client()


@pytest.fixture
def socketio_client(flask_app):
    """Cliente de prueba de Socket.IO"""
    return socketio.test_client(flask_app)


@pytest.fixture
def mock_detector():
    """Mock del detector de señas"""
    detector = Mock(spec=SignLanguageDetector)
    detector.predict.return_value = {
        'text': 'Hola',
        'confidence': 0.95,
        'class_id': 0,
        'language': 'LSC'
    }
    detector.get_stats.return_value = {
        'total_predictions': 10,
        'successful_predictions': 9,
        'failed_predictions': 1,
        'average_confidence': 0.88,
        'success_rate': 0.9
    }
    return detector


@pytest.fixture
def sample_frame():
    """Frame de prueba"""
    # Crear imagen de prueba (640x480, RGB)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_landmarks():
    """Landmarks de prueba"""
    # 21 puntos x 3 coordenadas = 63 valores
    landmarks = np.random.rand(63)
    return landmarks


@pytest.fixture
def sample_image_base64():
    """Imagen en base64 para pruebas"""
    import cv2
    import base64
    
    # Crear imagen simple
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Codificar a JPEG
    _, buffer = cv2.imencode('.jpg', img)
    
    # Convertir a base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64


@pytest.fixture
def temp_model_dir():
    """Directorio temporal para modelos"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_video_capture():
    """Mock de cv2.VideoCapture"""
    import cv2
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30
    }.get(prop, 0)
    
    return mock_cap


@pytest.fixture
def sample_training_data():
    """Datos de entrenamiento de prueba"""
    X_train = np.random.rand(100, 63)
    y_train = np.random.randint(0, 10, 100)
    X_val = np.random.rand(20, 63)
    y_val = np.random.randint(0, 10, 20)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Resetea variables de entorno antes de cada test"""
    yield
    # Cleanup después de cada test


@pytest.fixture
def mock_mediapipe_hands():
    """Mock de MediaPipe Hands"""
    from collections import namedtuple
    
    # Crear mock de landmarks
    Landmark = namedtuple('Landmark', ['x', 'y', 'z'])
    
    mock_landmarks = MagicMock()
    mock_landmarks.landmark = [
        Landmark(x=i*0.05, y=i*0.05, z=i*0.01) 
        for i in range(21)
    ]
    
    mock_results = MagicMock()
    mock_results.multi_hand_landmarks = [mock_landmarks]
    
    mock_hands = MagicMock()
    mock_hands.process.return_value = mock_results
    
    return mock_hands