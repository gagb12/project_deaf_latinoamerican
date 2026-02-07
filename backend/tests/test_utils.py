"""
Tests para utilidades
"""

import pytest
import numpy as np


class TestValidators:
    """Tests para validadores"""
    
    def test_validate_language(self):
        """Test validación de idioma"""
        from utils.validators import validate_language
        
        assert validate_language('LSC') == True
        assert validate_language('ASL') == True
        assert validate_language('INVALID') == False
    
    def test_validate_confidence(self):
        """Test validación de confianza"""
        from utils.validators import validate_confidence
        
        assert validate_confidence(0.5) == True
        assert validate_confidence(1.5) == False
        assert validate_confidence(-0.1) == False
    
    def test_validate_base64_image(self, sample_image_base64):
        """Test validación de imagen base64"""
        from utils.validators import validate_base64_image
        
        assert validate_base64_image(sample_image_base64) == True
        assert validate_base64_image('invalid') == False


class TestImageUtils:
    """Tests para utilidades de imágenes"""
    
    def test_preprocess_image(self, sample_frame):
        """Test preprocesamiento de imagen"""
        from utils.image_utils import preprocess_image
        
        processed = preprocess_image(sample_frame)
        
        assert processed is not None
    
    def test_resize_image(self, sample_frame):
        """Test redimensionar imagen"""
        from utils.image_utils import resize_image
        
        resized = resize_image(sample_frame, (320, 240))
        
        assert resized.shape[:2] == (240, 320)


class TestHelpers:
    """Tests para funciones helper"""
    
    def test_generate_session_id(self):
        """Test generar ID de sesión"""
        from utils.helpers import generate_session_id
        
        session_id = generate_session_id()
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    def test_format_timestamp(self):
        """Test formatear timestamp"""
        from utils.helpers import format_timestamp
        from datetime import datetime
        
        now = datetime.now()
        formatted = format_timestamp(now)
        
        assert isinstance(formatted, str)