from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from api.routes import api_bp
from api.websocket import setup_websocket
from config.settings import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Registrar blueprints
app.register_blueprint(api_bp, url_prefix='/api')

# Configurar WebSocket
setup_websocket(socketio)

@app.route('/')
def index():
    return {"message": "API Lengua de Señas - Activa"}

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)