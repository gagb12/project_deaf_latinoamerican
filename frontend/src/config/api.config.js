/**
 * Configuración de la API
 */

const API_CONFIG = {
  // URLs base
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:5000',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:5000',
  
  // Endpoints
  ENDPOINTS: {
    HEALTH: '/api/health',
    PREDICT: '/api/predict',
    HISTORY: '/api/history',
    STATS: '/api/stats',
    MODELS: '/api/models'
  },
  
  // Configuración de requests
  TIMEOUT: 30000, // 30 segundos
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 segundo
  
  // Headers por defecto
  DEFAULT_HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
};

export default API_CONFIG;