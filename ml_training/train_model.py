import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split

def create_model(input_shape, num_classes):
    """Crea modelo de red neuronal"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_lsc_model():
    """Entrena modelo para Lengua de Señas Colombiana"""
    # Aquí cargarías tu dataset LSC
    # X, y = load_lsc_dataset()
    
    # Ejemplo con datos sintéticos
    X = np.random.rand(1000, 63)  # 21 puntos * 3 coords
    y = np.random.randint(0, 10, 1000)  # 10 clases
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = create_model(input_shape=63, num_classes=10)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32
    )
    
    model.save('backend/models/trained_models/lsc_model.h5')
    print("✅ Modelo LSC guardado")
    
    return model, history

if __name__ == '__main__':
    train_lsc_model()