import React, { useState } from 'react';
import Camera from './components/Camera';
import SubtitleDisplay from './components/SubtitleDisplay';
import LanguageSelector from './components/LanguageSelector';
import './styles/App.css';

function App() {
  const [subtitle, setSubtitle] = useState('');
  const [language, setLanguage] = useState('LSC');
  const [confidence, setConfidence] = useState(0);

  const handlePrediction = (data) => {
    setSubtitle(data.text);
    setConfidence(data.confidence);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🤟 Traductor de Lengua de Señas</h1>
        <p>Colombiana (LSC) e Internacional (ASL)</p>
      </header>

      <main className="app-main">
        <LanguageSelector 
          language={language} 
          onLanguageChange={setLanguage} 
        />
        
        <Camera 
          language={language} 
          onPrediction={handlePrediction} 
        />
        
        <SubtitleDisplay 
          text={subtitle} 
          confidence={confidence}
          language={language}
        />
      </main>

      <footer className="app-footer">
        <p>Desarrollado para la comunidad sorda 💙</p>
      </footer>
    </div>
  );
}

export default App;