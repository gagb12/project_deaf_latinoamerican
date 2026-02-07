import React from 'react';
import '../styles/Subtitles.css';

const SubtitleDisplay = ({ text, confidence, language }) => {
  const getConfidenceColor = (conf) => {
    if (conf > 0.8) return '#4caf50';
    if (conf > 0.6) return '#ff9800';
    return '#f44336';
  };

  return (
    <div className="subtitle-container">
      <div className="subtitle-box">
        <span className="language-badge">
          {language === 'LSC' ? '🇨🇴 LSC' : '🌍 ASL'}
        </span>
        <h2 className="subtitle-text">{text || 'Esperando señas...'}</h2>
        {confidence > 0 && (
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{ 
                width: `${confidence * 100}%`,
                backgroundColor: getConfidenceColor(confidence)
              }}
            />
            <span className="confidence-text">
              {(confidence * 100).toFixed(1)}% confianza
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default SubtitleDisplay;