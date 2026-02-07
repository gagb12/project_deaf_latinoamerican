# 🤟 Traductor de Lengua de Señas LSC/ASL

Aplicación web para traducción en tiempo real de Lengua de Señas Colombiana (LSC) y American Sign Language (ASL) a texto.

## 🚀 Características

- ✅ Detección en tiempo real con cámara web
- ✅ Soporte para LSC y ASL
- ✅ Subtítulos con nivel de confianza
- ✅ Interfaz web responsive
- ✅ WebSocket para comunicación en tiempo real

## 📋 Requisitos

- Python 3.8+
- Node.js 16+
- Webcam

## 🛠️ Instalación

### Backend
```bash
cd backend
pip install -r requirements.txt
python server.py
----------------------------------------
Frontend

cd frontend
npm install
npm start
----------------------------------------

📖 Uso
Abre http://localhost:3000
Selecciona el idioma (LSC o ASL)
Permite acceso a la cámara
Realiza señas frente a la cámara
Ve los subtítulos en tiempo real
------------------------------------------

🤝 Contribuir
Las contribuciones son bienvenidas. Por favor:

Fork el proyecto
Crea una rama para tu feature
Commit tus cambios
Push a la rama
Abre un Pull Request
---------------------------------------------------
🚀 Comandos de Uso
Bash

# Iniciar con Docker Compose
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down

# Rebuild
docker-compose up --build

# Solo backend
docker-compose up backend

# Solo frontend
docker-compose up frontend
---------------------------------------------------------------

