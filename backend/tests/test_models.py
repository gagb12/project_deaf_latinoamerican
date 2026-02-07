"""
Tests para los modelos de ML
"""

import pytest
import numpy as np
import tempfile
import os


class TestSignLanguageDetector:
    """Tests para SignLanguageDetector"""
    
    def test_detector_initialization(self, temp_model_dir):
        """Test inicialización del detector"""
        from models.sign_detector import SignLanguageDetector
        
        # Crear un modelo dummy
        model_path = os.path.join(temp_model_dir, 'test_model.h5')
        
        # No debe fallar incluso sin modelo
        detector = SignLanguageDetector(model_path, language='LSC')
        
        assert detector.language == 'LSC'
        assert detector.labels is not None
    
    def test_load_labels_lsc(self):
        """Test carga de labels LSC"""
        from models.sign_detector import SignLanguageDetector
        
        detector = SignLanguageDetector('dummy_path.h5', language='LSC')
        labels = detector.labels
        
        assert isinstance(labels, dict)
        assert len(labels) > 0
        assert 0 in labels
    
    def test_load_labels_asl(self):
        """Test carga de labels ASL"""
        from models.sign_detector import SignLanguageDetector
        
        detector = SignLanguageDetector('dummy_path.h5', language='ASL')
        labels = detector.labels
        
        assert isinstance(labels, dict)
        assert len(labels) == 26  # A-Z
    
    def test_detect_hands_no_hands(self, sample_frame):
        """Test detección sin manos"""
        from models.sign_detector import SignLanguageDetector
        
        detector = SignLanguageDetector('dummy_path.h5')
        processed_frame, landmarks = detector.detect_hands(sample_frame)
        
        # Puede retornar None si no detecta manos
        assert processed_frame is not None
    
    def test_prediction_buffer(self):
        """Test buffer de predicciones"""
        from models.sign_detector import SignLanguageDetector
        
        detector = SignLanguageDetector('dummy_path.h5')
        
        # Agregar predicciones al buffer
        for i in range(10):
            detector._update_prediction_buffer(0, 0.9)
        
        assert len(detector.prediction_buffer) == detector.buffer_size
    
    def test_get_stats(self):
        """Test obtener estadísticas"""
        from models.sign_detector import SignLanguageDetector
        
        detector = SignLanguageDetector('dummy_path.h5')
        stats = detector.get_stats()
        
        assert 'total_predictions' in stats
        assert 'successful_predictions' in stats
        assert 'average_confidence' in stats


class TestPredictionModel:
    """Tests para PredictionModel"""
    
    def test_model_creation(self):
        """Test creación de modelo"""
        from models.prediction_model import PredictionModel
        
        model = PredictionModel(input_shape=63, num_classes=10)
        
        assert model.model is not None
        assert model.input_shape == 63
        assert model.num_classes == 10
    
    def test_model_compilation(self):
        """Test compilación de modelo"""
        from models.prediction_model import PredictionModel
        
        model = PredictionModel(input_shape=63, num_classes=10)
        model.compile_model()
        
        assert model.model.optimizer is not None
    
    def test_model_training(self, sample_training_data):
        """Test entrenamiento de modelo"""
        from models.prediction_model import PredictionModel
        
        model = PredictionModel(input_shape=63, num_classes=10)
        model.compile_model()
        
        history = model.train(
            sample_training_data['X_train'],
            sample_training_data['y_train'],
            sample_training_data['X_val'],
            sample_training_data['y_val'],
            epochs=2,
            batch_size=10
        )
        
        assert history is not None
        assert 'loss' in history.history
    
    def test_model_prediction(self, sample_training_data):
        """Test predicción de modelo"""
        from models.prediction_model import PredictionModel
        
        model = PredictionModel(input_shape=63, num_classes=10)
        model.compile_model()
        
        # Entrenar brevemente
        model.train(
            sample_training_data['X_train'],
            sample_training_data['y_train'],
            epochs=1,
            batch_size=10
        )
        
        # Predecir
        predictions = model.predict(sample_training_data['X_val'][:5])
        
        assert len(predictions) == 5
        assert all(0 <= p < 10 for p in predictions)
    
    def test_model_save_load(self, temp_model_dir):
        """Test guardar y cargar modelo"""
        from models.prediction_model import PredictionModel
        
        model_path = os.path.join(temp_model_dir, 'test_model.h5')
        
        # Crear y guardar modelo
        model = PredictionModel(input_shape=63, num_classes=10)
        model.compile_model()
        model.model_path = model_path
        model.save()
        
        assert os.path.exists(model_path)
        
        # Cargar modelo
        loaded_model = PredictionModel(model_path=model_path)
        assert loaded_model.model is not None


class TestHandTracker:
    """Tests para HandTracker"""
    
    def test_tracker_initialization(self):
        """Test inicialización del tracker"""
        from models.hand_tracker import HandTracker
        
        tracker = HandTracker()
        
        assert tracker.mp_hands is not None
        assert tracker.hands is not None
    
    def test_process_frame(self, sample_frame):
        """Test procesamiento de frame"""
        from models.hand_tracker import HandTracker
        
        tracker = HandTracker()
        processed_frame, hands_data = tracker.process_frame(sample_frame)
        
        assert processed_frame is not None
        assert isinstance(hands_data, list)
    
    def test_landmarks_extraction(self, mock_mediapipe_hands):
        """Test extracción de landmarks"""
        from models.hand_tracker import HandTracker
        
        tracker = HandTracker()
        
        # Usar mock
        tracker.hands = mock_mediapipe_hands
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed_frame, hands_data = tracker.process_frame(frame)
        
        assert len(hands_data) > 0


class TestVideoProcessor:
    """Tests para VideoProcessor"""
    
    def test_processor_initialization(self):
        """Test inicialización del procesador"""
        from models.video_processor import VideoProcessor
        
        processor = VideoProcessor(source=0)
        
        assert processor.width == 640
        assert processor.height == 480
    
    def test_resize_frame(self, sample_frame):
        """Test redimensionar frame"""
        from models.video_processor import VideoProcessor
        
        processor = VideoProcessor()
        resized = processor.resize_frame(sample_frame, width=320)
        
        assert resized.shape[1] == 320
    
    def test_flip_frame(self, sample_frame):
        """Test voltear frame"""
        from models.video_processor import VideoProcessor
        
        processor = VideoProcessor()
        flipped = processor.flip_frame(sample_frame)
        
        assert flipped.shape == sample_frame.shape
    
    def test_add_text(self, sample_frame):
        """Test agregar texto"""
        from models.video_processor import VideoProcessor
        
        processor = VideoProcessor()
        with_text = processor.add_text(sample_frame, "Test")
        
        assert with_text.shape == sample_frame.shape