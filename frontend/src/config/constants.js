/**
 * Constantes de la aplicación
 */

export const LANGUAGES = {
  LSC: {
    code: 'LSC',
    name: 'Lengua de Señas Colombiana',
    flag: '🇨🇴',
    description: 'Sistema de comunicación visual usado por la comunidad sorda en Colombia'
  },
  ASL: {
    code: 'ASL',
    name: 'American Sign Language',
    flag: '🌍',
    description: 'Sistema de comunicación visual usado principalmente en Estados Unidos'
  }
};

export const VIDEO_CONSTRAINTS = {
  width: { ideal: parseInt(process.env.REACT_APP_VIDEO_WIDTH) || 640 },
  height: { ideal: parseInt(process.env.REACT_APP_VIDEO_HEIGHT) || 480 },
  frameRate: { ideal: parseInt(process.env.REACT_APP_VIDEO_FPS) || 30 },
  facingMode: 'user'
};

export const CONFIDENCE_LEVELS = {
  HIGH: { min: 0.8, color: '#4CAF50', label: 'Alta' },
  MEDIUM: { min: 0.6, color: '#FF9800', label: 'Media' },
  LOW: { min: 0, color: '#F44336', label: 'Baja' }
};

export const ERROR_MESSAGES = {
  CAMERA_ACCESS: 'No se pudo acceder a la cámara',
  WEBSOCKET_ERROR: 'Error de conexión con el servidor',
  PREDICTION_ERROR: 'Error al procesar la señal',
  NETWORK_ERROR: 'Error de red. Verifica tu conexión'
};

export const FEATURES = {
  VOICE: process.env.REACT_APP_ENABLE_VOICE === 'true',
  HISTORY: process.env.REACT_APP_ENABLE_HISTORY === 'true',
  RECORDING: process.env.REACT_APP_ENABLE_RECORDING === 'true'
};

export const APP_INFO = {
  NAME: 'Traductor de Lengua de Señas',
  VERSION: '1.0.0',
  AUTHOR: 'Tu Nombre',
  REPO: 'https://github.com/tuusuario/lenguaje-senas-app'
};