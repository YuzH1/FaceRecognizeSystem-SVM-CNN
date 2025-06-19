import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .base_model import BaseFaceRecognitionModel


class FaceDataset(Dataset):
    """增强的PyTorch人脸数据集类"""
    
    def __init__(self, X, y, transform=None, is_training=False):
        """
        初始化人脸数据集
        
        Args:
            X: 人脸图像数据，格式为(N, C, H, W)或(N, H, W)或(N, H, W, C)
            y: 人脸标签
            transform: 数据增强转换
            is_training: 是否为训练模式
        """
        # 处理不同的输入格式
        if len(X.shape) == 4 and X.shape[-1] == 1:  # (N, H, W, 1)
            X = X.transpose(0, 3, 1, 2)  # 转换为(N, 1, H, W)
        elif len(X.shape) == 3:  # (N, H, W)
            X = np.expand_dims(X, axis=1)  # 转换为(N, 1, H, W)
        
        # 确保数据类型和范围正确
        X = X.astype(np.float32)
        if X.max() > 1.0:  # 如果不是[0-1]范围
            X = X / 255.0
            
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.transform = transform
        self.is_training = is_training
        
        # 基本图像转换
        self.basic_transform = transforms.Normalize((0.5,), (0.5,))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        # 应用数据增强
        if self.transform and self.is_training:
            # 转换为PIL图像
            x_np = x.numpy().transpose(1, 2, 0)  # CHW -> HWC
            x_np = np.squeeze(x_np)  # 移除通道维度(如果是单通道)
            x = self.transform(x_np)  # 应用转换，返回为Tensor
        else:
            # 仅应用基本的归一化
            x = self.basic_transform(x)
        
        # 返回数据和标签
        if self.y is not None:
            y = self.y[idx]
            return x, y
        else:
            return x


# 添加残差块
class ResidualBlock(nn.Module):
    """残差块实现"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                        stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


# 注意力机制
class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "注意力核大小必须是3或7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 生成空间注意力图
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        # 应用注意力
        return x * attention  # 乘法融合


class ImprovedFaceRecognitionNet(nn.Module):
    """改进的人脸识别网络"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10, 
                embedding_size: int = 512, dropout_rate: float = 0.3):
        super(ImprovedFaceRecognitionNet, self).__init__()
        
        # 输入层
        self.conv_init = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 特征提取层
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # 尺寸减半
            ResidualBlock(128, 128),
            SpatialAttention(kernel_size=7)  # 添加注意力
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),  # 尺寸减半
            ResidualBlock(256, 256)
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),  # 尺寸减半
            ResidualBlock(512, 512),
            SpatialAttention(kernel_size=7)  # 添加注意力
        )
        
        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征嵌入层
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_size, num_classes)
        )
    
    def forward_features(self, x):
        """特征提取"""
        x = self.conv_init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        return self.embedding_layer(x)  # 返回特征嵌入
    
    def forward(self, x):
        """前向传播"""
        features = self.forward_features(x)
        return self.classifier(features)


class PyTorchCNNFaceRecognitionModel(BaseFaceRecognitionModel):
    """基于PyTorch CNN的改进人脸识别模型"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (64, 64, 1), 
                learning_rate: float = 0.001, batch_size: int = 32, 
                dropout_rate: float = 0.3, embedding_size: int = 512,
                device: Optional[str] = None):
        """
        初始化模型
        
        Args:
            input_shape: 输入图像形状 (高度, 宽度, 通道数)
            learning_rate: 学习率
            batch_size: 批大小
            dropout_rate: Dropout比率
            embedding_size: 特征嵌入维度
            device: 设备(cpu/cuda)
        """
        super().__init__()
        
        # 设置设备
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"使用设备: {self.device}")
        
        # 模型参数
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.embedding_size = embedding_size
        
        # 数据变换
        self.transform = None
        self.label_encoder = None
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化PyTorch模型"""
        # 确定输入通道数
        input_channels = self.input_shape[-1] if len(self.input_shape) == 3 else 1
        
        # 创建数据增强
        self._create_transforms()
        
        # 等待初始化后确定类别数
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
    
    def _create_transforms(self):
        """创建数据增强转换"""
        # 训练时使用的数据增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Any:
        """
        准备输入数据
        
        Args:
            X: 人脸图像数据
            y: 可选的标签数据
            
        Returns:
            处理后的数据
        """
        # 如果没有模型，则需要初始化
        if self.model is None and y is not None:
            # 初始化标签编码器
            self.label_encoder = LabelEncoder()
            encoded_y = self.label_encoder.fit_transform(y)
            num_classes = len(self.label_encoder.classes_)
            
            # 现在可以初始化模型
            input_channels = 1 if len(X.shape) == 3 or X.shape[-1] == 1 else X.shape[-1]
            self.model = ImprovedFaceRecognitionNet(
                input_channels=input_channels, 
                num_classes=num_classes,
                embedding_size=self.embedding_size,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            # 初始化优化器和损失函数
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-5  # L2正则化
            )
            
            # 带标签平滑的交叉熵损失
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            print(f"初始化模型完成，类别数: {num_classes}")
            return X, encoded_y
        
        # 已初始化的情况
        if y is not None and self.label_encoder is not None:
            encoded_y = self.label_encoder.transform(y)
            return X, encoded_y
        return X
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             epochs: int = 50, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X_train: 训练图像
            y_train: 训练标签
            X_val: 可选的验证图像
            y_val: 可选的验证标签
            epochs: 训练轮次
            batch_size: 可选的批大小，覆盖初始化时设置的值
            
        Returns:
            包含训练历史的字典
        """
        # 准备数据
        X_train_prep, y_train_encoded = self._prepare_data(X_train, y_train)
        
        # 设置批大小
        if batch_size is not None:
            self.batch_size = batch_size
        
        # 创建数据加载器
        train_dataset = FaceDataset(X_train_prep, y_train_encoded, 
                                   transform=self.transform, is_training=True)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=2, pin_memory=True
        )
        
        # 如果有验证集
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_prep, y_val_encoded = self._prepare_data(X_val, y_val)
            val_dataset = FaceDataset(X_val_prep, y_val_encoded)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=2, pin_memory=True
            )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, 
            min_lr=1e-6, verbose=True
        )
        
        # 启用自动混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        
        # 训练循环
        print(f"开始训练，共{epochs}轮...")
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(train_loader, scaler)
            
            # 验证
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            if val_loader is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # 输出进度
            if epoch % 5 == 0 or epoch == epochs - 1:
                val_info = f", 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}" if val_loader else ""
                print(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}{val_info}")
        
        # 训练完成，恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"训练完成，恢复最佳模型 (验证准确率: {best_val_acc:.4f})")
        else:
            print("训练完成")
        
        # 设置为已训练状态
        self.is_trained = True
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, 
                    scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            scaler: 可选的混合精度训练缩放器
            
        Returns:
            epoch平均损失和准确率
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 使用混合精度训练
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # 普通训练
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 计算统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # 计算epoch平均值
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证损失和准确率
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # 计算平均值
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 输入图像
            
        Returns:
            预测的类别标签
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        
        # 准备数据
        X_prep = self._prepare_data(X)
        
        # 创建数据集和加载器
        dataset = FaceDataset(X_prep, y=None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # 预测
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                predictions.append(predicted.cpu().numpy())
        
        # 合并批次结果
        predicted_encoded = np.concatenate(predictions)
        
        # 转换回原始标签
        if self.label_encoder:
            return self.label_encoder.inverse_transform(predicted_encoded)
        
        return predicted_encoded
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入图像
            
        Returns:
            每个类别的概率
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        
        # 准备数据
        X_prep = self._prepare_data(X)
        
        # 创建数据集和加载器
        dataset = FaceDataset(X_prep, y=None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # 预测
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                probabilities.append(probs.cpu().numpy())
        
        # 合并批次结果
        return np.concatenate(probabilities)
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        提取特征嵌入
        
        Args:
            X: 输入图像
            
        Returns:
            特征嵌入向量
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        
        # 准备数据
        X_prep = self._prepare_data(X)
        
        # 创建数据集和加载器
        dataset = FaceDataset(X_prep, y=None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # 提取特征
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(self.device)
                # 使用模型的特征提取部分
                feats = self.model.forward_features(inputs)
                features.append(feats.cpu().numpy())
        
        # 合并批次结果
        return np.concatenate(features)
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法保存")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型和额外信息
        save_dict = {
            'model_state': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'history': self.history,
            'input_shape': self.input_shape,
            'embedding_size': self.embedding_size,
            'dropout_rate': self.dropout_rate,
        }
        
        torch.save(save_dict, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载模型
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 恢复模型配置
        self.input_shape = checkpoint.get('input_shape', self.input_shape)
        self.embedding_size = checkpoint.get('embedding_size', self.embedding_size)
        self.dropout_rate = checkpoint.get('dropout_rate', self.dropout_rate)
        self.label_encoder = checkpoint.get('label_encoder')
        self.history = checkpoint.get('history', self.history)
        
        # 初始化模型
        input_channels = self.input_shape[-1] if len(self.input_shape) == 3 else 1
        num_classes = len(self.label_encoder.classes_) if self.label_encoder else 1
        
        self.model = ImprovedFaceRecognitionNet(
            input_channels=input_channels,
            num_classes=num_classes,
            embedding_size=self.embedding_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state'])
        
        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 设置为已训练状态
        self.is_trained = True
        
        print(f"模型已从 {filepath} 加载，类别数: {num_classes}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        if not self.history['train_loss']:
            print("没有训练历史可以绘制")
            return
        
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='训练准确率')
        if self.history['val_acc']:
            plt.plot(self.history['val_acc'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self):
        """获取模型摘要信息"""
        if self.model is None:
            return "模型未初始化"
        
        model_info = []
        model_info.append(f"模型类型: {self.model.__class__.__name__}")
        model_info.append(f"输入形状: {self.input_shape}")
        
        if self.label_encoder is not None:
            model_info.append(f"类别数: {len(self.label_encoder.classes_)}")
            model_info.append(f"类别: {', '.join(self.label_encoder.classes_)}")
        
        model_info.append(f"特征嵌入大小: {self.embedding_size}")
        model_info.append(f"Dropout率: {self.dropout_rate}")
        model_info.append(f"训练状态: {'已训练' if self.is_trained else '未训练'}")
        model_info.append(f"设备: {self.device}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_info.append(f"总参数量: {total_params:,}")
        model_info.append(f"可训练参数: {trainable_params:,}")
        
        return '\n'.join(model_info)
    
    def debug_prediction(self, X: np.ndarray, top_k: int = 3):
        """
        调试预测过程
        
        Args:
            X: 输入图像
            top_k: 显示的前K个预测结果
        """
        if not self.is_trained:
            print("模型未训练，无法调试")
            return
        
        # 获取单个图像
        single_image = X[0:1] if len(X.shape) == 4 else X.reshape(1, *X.shape)
        
        # 输出输入图像信息
        print(f"输入图像形状: {single_image.shape}, 类型: {single_image.dtype}")
        print(f"值范围: [{np.min(single_image):.3f}, {np.max(single_image):.3f}], 均值: {np.mean(single_image):.3f}")
        
        # 准备数据
        X_prep = self._prepare_data(single_image)
        dataset = FaceDataset(X_prep, y=None)
        
        # 获取处理后的图像
        processed_img = dataset[0]
        if isinstance(processed_img, tuple):
            processed_img = processed_img[0]  # 获取图像部分
            
        # 转换为numpy
        if isinstance(processed_img, torch.Tensor):
            processed_img = processed_img.numpy()
        
        # 预测
        self.model.eval()
        
        # 转换为tensor并移动到设备
        if isinstance(processed_img, np.ndarray):
            input_tensor = torch.FloatTensor(processed_img).unsqueeze(0).to(self.device)
        else:
            input_tensor = processed_img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取特征和输出
            features = self.model.forward_features(input_tensor)
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            
            # 获取前K个预测
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
            
        # 显示输入图像
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        if len(single_image.shape) == 4 and single_image.shape[-1] == 1:
            img_display = np.squeeze(single_image[0])
        elif len(single_image.shape) == 4 and single_image.shape[-1] == 3:
            img_display = single_image[0]
        else:
            img_display = np.squeeze(single_image)
            
        plt.imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
        plt.title("输入图像")
        plt.axis('off')
        
        # 显示预测结果
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(top_indices))
        
        # 获取类别名称
        if self.label_encoder:
            labels = [self.label_encoder.classes_[idx] for idx in top_indices.cpu().numpy()]
        else:
            labels = [f"类别 {idx}" for idx in top_indices.cpu().numpy()]
        
        # 绘制条形图
        bars = plt.barh(y_pos, top_probs.cpu().numpy())
        plt.yticks(y_pos, labels)
        plt.xlabel('预测概率')
        plt.title(f'前{len(top_indices)}个预测结果')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{top_probs[i]:.4f}', 
                va='center'
            )
        
        plt.tight_layout()
        plt.show()
        
        # 输出特征信息
        print(f"特征嵌入形状: {features.shape}")
        print(f"特征统计: 均值={torch.mean(features).item():.4f}, 标准差={torch.std(features).item():.4f}")
