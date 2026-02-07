"""
Configuración de CORS
"""

from flask_cors import CORS


class CORSConfig:
    """Configuración de CORS"""
    
    @staticmethod
    def init_app(app):
        """Inicializar CORS"""
        
        # Configuración básica
        cors_config = {
            'origins': app.config.get('CORS_ORIGINS', '*'),
            'methods': app.config.get('CORS_METHODS', ['GET', 'POST', 'PUT', 'DELETE']),
            'allow_headers': app.config.get('CORS_ALLOW_HEADERS', ['Content-Type', 'Authorization']),
            'supports_credentials': True,
            'max_age': 3600
        }
        
        CORS(app, resources={r"/*": cors_config})
        
        print(f"✅ CORS configurado para: {cors_config['origins']}")