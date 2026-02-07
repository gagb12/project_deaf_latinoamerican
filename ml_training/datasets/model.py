"""
model.py - Modelo de reconocimiento de lengua de señas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .config import ModelConfig, config

import logging
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Codificación posicional para el Transformer"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignLanguageTransformer(nn.Module):
    """
    Modelo Transformer para reconocimiento de señas.
    Procesa secuencias temporales de features de landmarks.
    """

    def __init__(
        self,
        input_features: int = 258,
        num_classes: int = 100,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.3,
        sequence_length: int = 30
    ):
        super().__init__()

        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Proyección de entrada
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Codificación posicional
        self.positional_encoding = PositionalEncoding(
            hidden_size, max_len=sequence_length, dropout=dropout
        )

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Attention pooling (en vez de solo el último token)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Cabeza de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Inicialización de pesos
        self._init_weights()

        logger.info(
            f"SignLanguageTransformer: {input_features} features → "
            f"{num_classes} clases, {self.count_parameters()} parámetros"
        )

    def _init_weights(self):
        """Inicialización Xavier"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_features)
            mask: Máscara de padding opcional

        Returns:
            (batch, num_classes) logits
        """
        # Proyectar input
        x = self.input_projection(x)   # (B, T, hidden)

        # Añadir posición temporal
        x = self.positional_encoding(x)

        # Transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Attention pooling
        attention_weights = self.attention_pool(x)   # (B, T, 1)
        x = torch.sum(x * attention_weights, dim=1)  # (B, hidden)

        # Clasificar
        logits = self.classifier(x)  # (B, num_classes)

        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicción con probabilidades

        Returns:
            (predicted_class, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
            predicted = torch.argmax(probabilities, dim=-1)
        return predicted, probabilities

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SignLanguageLSTM(nn.Module):
    """
    Alternativa LSTM (más ligera, buena para dispositivos móviles)
    """

    def __init__(
        self,
        input_features: int = 258,
        num_classes: int = 100,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Normalización de entrada
        self.input_norm = nn.LayerNorm(input_features)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Usar el último timestep
        if self.bidirectional:
            last_hidden = torch.cat(
                (hidden[-2], hidden[-1]), dim=-1
            )
        else:
            last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)
        return logits


class SignLanguageHybrid(nn.Module):
    """
    Modelo híbrido: CNN (features espaciales) + Transformer (temporales)
    Mejor rendimiento general.
    """

    def __init__(
        self,
        input_features: int = 258,
        num_classes: int = 100,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()

        # CNN 1D para features espaciales locales
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(input_features, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer para relaciones temporales
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Clasificador
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Pool temporal
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, Features)
        # CNN espera (B, Features, T)
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.spatial_cnn(x_cnn)
        x = x_cnn.permute(0, 2, 1)  # Back to (B, T, H)

        x = self.pos_encoding(x)
        x = self.transformer(x)

        # Clasificar
        x = x.permute(0, 2, 1)  # (B, H, T)
        logits = self.classifier(x)

        return logits


def create_model(
    model_config: Optional[ModelConfig] = None,
    num_classes: int = 100
) -> nn.Module:
    """
    Factory function para crear el modelo apropiado

    Args:
        model_config: Configuración del modelo
        num_classes: Número de clases (señas)

    Returns:
        Modelo PyTorch
    """
    cfg = model_config or config.model

    models = {
        "transformer": SignLanguageTransformer,
        "lstm": SignLanguageLSTM,
        "hybrid": SignLanguageHybrid,
    }

    ModelClass = models.get(cfg.architecture, SignLanguageTransformer)

    if cfg.architecture == "lstm":
        return ModelClass(
            input_features=cfg.input_features,
            num_classes=num_classes,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    else:
        return ModelClass(
            input_features=cfg.input_features,
            num_classes=num_classes,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )