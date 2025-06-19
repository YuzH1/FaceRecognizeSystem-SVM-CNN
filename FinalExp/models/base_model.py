# models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

class BaseFaceRecognitionModel(ABC):
    """人脸识别模型基类"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """保存模型"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """加载模型"""
        pass