/**
 * Configuración de WebSocket
 */

const WS_CONFIG = {
  // URL de conexión
  url: process.env.REACT_APP_WS_URL || 'ws://localhost:5000',
  
  // Opciones de Socket.IO
  options: {
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    autoConnect: true
  },
  
  // Eventos
  EVENTS: {
    CONNECT: 'connect',
    DISCONNECT: 'disconnect',
    VIDEO_FRAME: 'video_frame',
    PREDICTION: 'prediction',
    ERROR: 'error',
    STATUS: 'status',
    LANGUAGE_CHANGE: 'language_change'
  },
  
  // Configuración de envío de frames
  FRAME_CONFIG: {
    sendInterval: 100, // ms (10 FPS)
    quality: 0.8, // Calidad JPEG (0-1)
    maxSize: 1024 * 1024 // 1MB max
  }
};

export default WS_CONFIG;