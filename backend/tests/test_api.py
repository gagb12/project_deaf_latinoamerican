"""
Tests para endpoints REST de la API
"""

import pytest
import json


class TestHealthEndpoint:
    """Tests para el endpoint /api/health"""
    
    def test_health_check_success(self, client):
        """Test que health check retorna 200"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_health_check_returns_version(self, client):
        """Test que health check retorna versión"""
        response = client.get('/api/health')
        data = json.loads(response.data)
        
        assert 'version' in data
        assert isinstance(data['version'], str)
    
    def test_health_check_returns_models_status(self, client):
        """Test que health check retorna estado de modelos"""
        response = client.get('/api/health')
        data = json.loads(response.data)
        
        assert 'models' in data
        assert isinstance(data['models'], dict)


class TestModelsEndpoint:
    """Tests para el endpoint /api/models"""
    
    def test_get_models_list(self, client):
        """Test obtener lista de modelos"""
        response = client.get('/api/models')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'models' in data
        assert isinstance(data['models'], list)
    
    def test_models_have_required_fields(self, client):
        """Test que modelos tienen campos requeridos"""
        response = client.get('/api/models')
        data = json.loads(response.data)
        
        if len(data['models']) > 0:
            model = data['models'][0]
            required_fields = ['language', 'name', 'version']
            
            for field in required_fields:
                assert field in model


class TestPredictEndpoint:
    """Tests para el endpoint /api/predict"""
    
    def test_predict_requires_image(self, client):
        """Test que predict requiere imagen"""
        response = client.post(
            '/api/predict',
            json={'language': 'LSC'}
        )
        
        assert response.status_code == 400
    
    def test_predict_requires_language(self, client, sample_image_base64):
        """Test que predict requiere idioma"""
        response = client.post(
            '/api/predict',
            json={'image': sample_image_base64}
        )
        
        # Debería usar idioma por defecto o retornar error
        assert response.status_code in [200, 400]
    
    def test_predict_with_valid_data(self, client, sample_image_base64):
        """Test predicción con datos válidos"""
        response = client.post(
            '/api/predict',
            json={
                'image': sample_image_base64,
                'language': 'LSC'
            }
        )
        
        # Puede ser 200 o 422 dependiendo de si detecta manos
        assert response.status_code in [200, 422]
    
    def test_predict_invalid_language(self, client, sample_image_base64):
        """Test predicción con idioma inválido"""
        response = client.post(
            '/api/predict',
            json={
                'image': sample_image_base64,
                'language': 'INVALID'
            }
        )
        
        assert response.status_code == 400
    
    def test_predict_response_structure(self, client, sample_image_base64, mocker):
        """Test estructura de respuesta de predicción"""
        # Mock del detector para retornar predicción válida
        mock_predict = mocker.patch(
            'models.sign_detector.SignLanguageDetector.predict',
            return_value={
                'text': 'Hola',
                'confidence': 0.95,
                'class_id': 0,
                'language': 'LSC'
            }
        )
        
        response = client.post(
            '/api/predict',
            json={
                'image': sample_image_base64,
                'language': 'LSC'
            }
        )
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'prediction' in data
            assert 'text' in data['prediction']
            assert 'confidence' in data['prediction']


class TestHistoryEndpoint:
    """Tests para el endpoint /api/history"""
    
    def test_get_history(self, client):
        """Test obtener historial"""
        response = client.get('/api/history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'results' in data
    
    def test_history_with_limit(self, client):
        """Test historial con límite"""
        response = client.get('/api/history?limit=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'limit' in data
        assert data['limit'] == 5
    
    def test_history_with_language_filter(self, client):
        """Test historial filtrado por idioma"""
        response = client.get('/api/history?language=LSC')
        
        assert response.status_code == 200
    
    def test_history_pagination(self, client):
        """Test paginación del historial"""
        response = client.get('/api/history?limit=10&offset=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'offset' in data


class TestStatsEndpoint:
    """Tests para el endpoint /api/stats"""
    
    def test_get_stats(self, client):
        """Test obtener estadísticas"""
        response = client.get('/api/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        expected_fields = [
            'total_predictions',
            'predictions_by_language',
            'average_confidence'
        ]
        
        for field in expected_fields:
            assert field in data


class TestCORSHeaders:
    """Tests para headers CORS"""
    
    def test_cors_headers_present(self, client):
        """Test que headers CORS están presentes"""
        response = client.get('/api/health')
        
        assert 'Access-Control-Allow-Origin' in response.headers
    
    def test_cors_options_request(self, client):
        """Test request OPTIONS para CORS"""
        response = client.options('/api/predict')
        
        assert response.status_code in [200, 204]


class TestErrorHandling:
    """Tests para manejo de errores"""
    
    def test_404_error(self, client):
        """Test error 404"""
        response = client.get('/api/nonexistent')
        
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, client):
        """Test error 405"""
        response = client.delete('/api/health')
        
        assert response.status_code == 405
    
    def test_invalid_json(self, client):
        """Test JSON inválido"""
        response = client.post(
            '/api/predict',
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code == 400