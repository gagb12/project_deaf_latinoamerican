"""
trainer.py - Entrenamiento del modelo de señas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging
from tqdm import tqdm

from .model import create_model
from .config import config, ModelConfig

logger = logging.getLogger(__name__)


class SignLanguageDataset(Dataset):
    """Dataset de lengua de señas"""

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 30,
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.augment = augment

        # Cargar datos
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._load_data()

    def _load_data(self):
        """Carga todos los datos del directorio"""
        # Encontrar todas las clases (subdirectorios)
        class_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and len(list(d.glob("*.npy"))) > 0
        ])

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Cargar todas las muestras
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            for npy_file in class_dir.glob("*.npy"):
                try:
                    data = np.load(npy_file)

                    # Ajustar longitud de secuencia
                    if len(data) > self.sequence_length:
                        data = data[:self.sequence_length]
                    elif len(data) < self.sequence_length:
                        padding = np.zeros(
                            (self.sequence_length - len(data), data.shape[1]),
                            dtype=np.float32
                        )
                        data = np.vstack([data, padding])

                    self.samples.append((data, class_idx))

                except Exception as e:
                    logger.warning(f"Error cargando {npy_file}: {e}")

        logger.info(
            f"Dataset cargado: {len(self.samples)} muestras, "
            f"{len(self.classes)} clases"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]

        if self.augment:
            data = self._augment(data)

        return (
            torch.FloatTensor(data),
            torch.LongTensor([label]).squeeze()
        )

    def _augment(self, data: np.ndarray) -> np.ndarray:
        """Aumentación de datos"""
        augmented = data.copy()

        # 1. Ruido gaussiano
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented += noise.astype(np.float32)

        # 2. Escalamiento temporal (velocidad)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            indices = np.linspace(
                0, len(augmented) - 1,
                int(len(augmented) * scale)
            ).astype(int)
            indices = np.clip(indices, 0, len(augmented) - 1)
            augmented = augmented[indices]

            # Re-ajustar longitud
            if len(augmented) > self.sequence_length:
                augmented = augmented[:self.sequence_length]
            elif len(augmented) < self.sequence_length:
                pad = np.zeros(
                    (self.sequence_length - len(augmented), augmented.shape[1]),
                    dtype=np.float32
                )
                augmented = np.vstack([augmented, pad])

        # 3. Espejo horizontal (invertir x)
        if np.random.random() < 0.3:
            # Invertir coordenadas x (cada 3 valores en los landmarks)
            for i in range(0, augmented.shape[1], 3):
                if i < augmented.shape[1]:
                    augmented[:, i] = 1.0 - augmented[:, i]

        # 4. Dropout de frames
        if np.random.random() < 0.2:
            num_drop = np.random.randint(1, 4)
            drop_indices = np.random.choice(
                len(augmented), num_drop, replace=False
            )
            augmented[drop_indices] = 0.0

        return augmented


class SignLanguageTrainer:
    """Entrenador del modelo de señas"""

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "./data/models",
        model_config: Optional[ModelConfig] = None,
        device: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_config = model_config or config.model
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        logger.info(f"Usando dispositivo: {self.device}")

        # Dataset
        self.dataset = SignLanguageDataset(
            data_dir=data_dir,
            sequence_length=self.model_config.sequence_length,
            augment=True
        )

        # Split train/val/test (70/15/15)
        total = len(self.dataset)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False
        )

        # Modelo
        # Detectar input features del dataset
        sample_data, _ = self.dataset[0]
        actual_features = sample_data.shape[1]
        self.model_config.input_features = actual_features

        self.model = create_model(
            self.model_config,
            num_classes=len(self.dataset.classes)
        ).to(self.device)

        # Optimizador y scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=0.01
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Loss con pesos de clase (para clases desbalanceadas)
        class_counts = self._count_classes()
        weights = 1.0 / torch.FloatTensor(
            [class_counts.get(i, 1) for i in range(len(self.dataset.classes))]
        )
        weights = weights / weights.sum() * len(self.dataset.classes)
        self.criterion = nn.CrossEntropyLoss(
            weight=weights.to(self.device),
            label_smoothing=0.1
        )

        # Historial
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }

        logger.info(
            f"Trainer inicializado:\n"
            f"  Clases: {len(self.dataset.classes)}\n"
            f"  Train: {len(self.train_dataset)}\n"
            f"  Val: {len(self.val_dataset)}\n"
            f"  Test: {len(self.test_dataset)}\n"
            f"  Features: {actual_features}\n"
            f"  Modelo: {self.model_config.architecture}"
        )

    def _count_classes(self) -> Dict[int, int]:
        """Cuenta muestras por clase"""
        counts = {}
        for _, label in self.dataset.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def train(self) -> Dict:
        """Entrena el modelo completo"""
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(
            f"Iniciando entrenamiento: {self.model_config.epochs} epochs"
        )

        for epoch in range(self.model_config.epochs):
            # Train
            train_loss, train_acc = self._train_epoch()

            # Validate
            val_loss, val_acc = self._validate()

            # Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Log
            logger.info(
                f"Epoch {epoch+1}/{self.model_config.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1%} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1%} | "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.model_config.patience:
                    logger.info(
                        f"Early stopping en epoch {epoch+1}"
                    )
                    break

        # Evaluación final
        test_results = self.evaluate()
        self._save_training_report(test_results)

        return test_results

    def _train_epoch(self) -> Tuple[float, float]:
        """Entrena una epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_data, batch_labels in tqdm(
            self.train_loader, desc="Training", leave=False
        ):
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def _validate(self) -> Tuple[float, float]:
        """Valida el modelo"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in self.val_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def evaluate(self) -> Dict:
        """Evaluación completa en test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data, batch_labels in self.test_loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Métricas
        accuracy = np.mean(all_preds == all_labels)

        # Accuracy por clase
        class_accuracy = {}
        for idx, class_name in enumerate(self.dataset.classes):
            mask = all_labels == idx
            if mask.sum() > 0:
                class_accuracy[class_name] = float(
                    np.mean(all_preds[mask] == all_labels[mask])
                )

        results = {
            'overall_accuracy': float(accuracy),
            'class_accuracy': class_accuracy,
            'num_classes': len(self.dataset.classes),
            'num_test_samples': len(all_labels),
            'classes': self.dataset.classes,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Test Accuracy: {accuracy:.1%}")
        return results

    def _save_checkpoint(
        self, epoch: int, val_loss: float,
        val_acc: float, is_best: bool = False
    ):
        """Guarda checkpoint del modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'classes': self.dataset.classes,
            'class_to_idx': self.dataset.class_to_idx,
            'model_config': {
                'architecture': self.model_config.architecture,
                'input_features': self.model_config.input_features,
                'hidden_size': self.model_config.hidden_size,
                'num_layers': self.model_config.num_layers,
                'num_heads': self.model_config.num_heads,
                'dropout': self.model_config.dropout,
                'num_classes': len(self.dataset.classes),
            },
            'language': config.active_language,
        }

        if is_best:
            path = self.output_dir / f"{config.active_language.lower()}_best.pth"
            torch.save(checkpoint, path)
            logger.info(f"Mejor modelo guardado: {path}")

    def _save_training_report(self, test_results: Dict):
        """Guarda reporte de entrenamiento"""
        report = {
            'test_results': test_results,
            'history': self.history,
            'config': {
                'architecture': self.model_config.architecture,
                'sequence_length': self.model_config.sequence_length,
                'hidden_size': self.model_config.hidden_size,
                'learning_rate': self.model_config.learning_rate,
                'batch_size': self.model_config.batch_size,
                'language': config.active_language,
            }
        }

        report_path = self.output_dir / f"training_report_{config.active_language.lower()}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Reporte guardado: {report_path}")