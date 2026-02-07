import React from 'react';

const LanguageSelector = ({ language, onLanguageChange }) => {
  return (
    <div className="language-selector">
      <label>Seleccionar idioma de señas:</label>
      <div className="language-buttons">
        <button
          className={language === 'LSC' ? 'active' : ''}
          onClick={() => onLanguageChange('LSC')}
        >
          🇨🇴 Lengua de Señas Colombiana (LSC)
        </button>
        <button
          className={language === 'ASL' ? 'active' : ''}
          onClick={() => onLanguageChange('ASL')}
        >
          🌍 American Sign Language (ASL)
        </button>
      </div>
    </div>
  );
};

export default LanguageSelector;