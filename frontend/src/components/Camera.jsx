import React, { useRef, useEffect, useState } from 'react';
import io from 'socket.io-client';
import '../styles/Camera.css';

const Camera = ({ language, onPrediction }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [socket, setSocket] = useState(null);
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    // Conectar WebSocket
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('prediction', (data) => {
      onPrediction(data);
    });

    return () => newSocket.close();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsActive(true);
        captureFrames();
      }
    } catch (err) {
      console.error('Error accediendo a la cámara:', err);
      alert('No se pudo acceder a la cámara');
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject;
    const tracks = stream?.getTracks();
    tracks?.forEach(track => track.stop());
    setIsActive(false);
  };

  const captureFrames = () => {
    const interval = setInterval(() => {
      if (!isActive || !socket) return;

      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      // Enviar frame al servidor
      canvas.toBlob((blob) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64data = reader.result.split(',')[1];
          socket.emit('video_frame', { 
            image: base64data,
            language: language 
          });
        };
        reader.readAsDataURL(blob);
      }, 'image/jpeg', 0.8);
    }, 100); // 10 FPS

    return () => clearInterval(interval);
  };

  return (
    <div className="camera-container">
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline
        className="video-feed"
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      
      <div className="camera-controls">
        {!isActive ? (
          <button onClick={startCamera} className="btn-start">
            📹 Iniciar Cámara
          </button>
        ) : (
          <button onClick={stopCamera} className="btn-stop">
            ⏹️ Detener
          </button>
        )}
      </div>
    </div>
  );
};

export default Camera;