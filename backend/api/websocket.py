from flask_socketio import emit
import cv2
import base64
import numpy as np
from models.sign_detector import SignLanguageDetector

# Instancias de detectores
detector_lsc = None
detector_asl = None

def setup_websocket(socketio):
    global detector_lsc, detector_asl
    
    # Cargar modelos al iniciar
    try:
        detector_lsc = SignLanguageDetector(
            'models/trained_models/lsc_model.h5', 
            language='LSC'
        )
        detector_asl = SignLanguageDetector(
            'models/trained_models/asl_model.h5', 
            language='ASL'
        )
    except:
        print("⚠️ Modelos no encontrados. Entrenamiento necesario.")
    
    @socketio.on('connect')
    def handle_connect():
        print('✅ Cliente conectado')
        emit('response', {'message': 'Conectado al servidor'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print('❌ Cliente desconectado')
    
    @socketio.on('video_frame')
    def handle_video_frame(data):
        """Procesa frames de video en tiempo real"""
        try:
            # Decodificar imagen
            img_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Seleccionar detector según idioma
            language = data.get('language', 'LSC')
            detector = detector_lsc if language == 'LSC' else detector_asl
            
            if detector:
                # Predecir
                prediction = detector.predict(frame)
                
                if prediction:
                    emit('prediction', {
                        'text': prediction['text'],
                        'confidence': prediction['confidence'],
                        'language': language
                    })
        except Exception as e:
            print(f"Error procesando frame: {e}")
            emit('error', {'message': str(e)})