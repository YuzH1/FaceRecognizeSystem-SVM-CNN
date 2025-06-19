# models/svm_model.py
import numpy as np
import joblib
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict, Any, List, Tuple, Optional
import warnings
from .base_model import BaseFaceRecognitionModel

class SVMFaceRecognitionModel(BaseFaceRecognitionModel):
    """增强版基于SVM的人脸识别模型"""
    
    def __init__(self, feature_method: str = 'hog', n_components: int = 150, 
                kernel: str = 'rbf', C: float = 10.0, gamma: str = 'scale',
                use_ensemble: bool = False, class_weight: str = 'balanced'):
        """
        初始化SVM模型
        
        Args:
            feature_method: 特征提取方法，'pixel'(原始像素), 'hog', 'lbp', 'hog+lbp'
            n_components: PCA保留的主成分数量
            kernel: SVM核函数类型
            C: SVM正则化参数
            gamma: RBF核参数
            use_ensemble: 是否使用集成学习
            class_weight: 类别权重，处理不平衡数据集
        """
        super().__init__()
        self.feature_method = feature_method
        self.n_components = n_components
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.use_ensemble = use_ensemble
        self.class_weight = class_weight
        
        # 初始化组件
        self.scaler = StandardScaler()
        self.pca = None  # 将在训练时根据数据初始化
        self.svm = None  # 将在训练时根据数据初始化
        self.label_encoder = LabelEncoder()
        self.pipeline = None
        self.ensemble = None
        self.best_params = None
        self.feature_cache = {}  # 缓存提取的特征
        
        # 初始化HOG参数
        self.hog_params = {
            'winSize': (64, 64),
            'blockSize': (16, 16),
            'blockStride': (8, 8),
            'cellSize': (8, 8),
            'nbins': 9
        }
        
        # 初始化LBP参数
        self.lbp_params = {
            'radius': 2,
            'n_points': 8 * 2,
            'grid_x': 8,
            'grid_y': 8
        }
    
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        从图像中提取特征
        
        Args:
            X: 图像数据，形状为(n_samples, height, width)或(n_samples, height, width, channels)
            
        Returns:
            特征矩阵，形状为(n_samples, n_features)
        """
        # 检查缓存
        cache_key = f"{id(X)}_{self.feature_method}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # 图像预处理和格式转换
        if X.ndim == 4 and X.shape[3] == 1:  # (n_samples, height, width, 1)
            X_processed = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        elif X.ndim == 4 and X.shape[3] == 3:  # 彩色图像
            # 转换为灰度图
            X_processed = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
            for i in range(X.shape[0]):
                if X.dtype == np.float32 and X[i].max() <= 1.0:
                    img = (X[i] * 255).astype(np.uint8)
                else:
                    img = X[i].astype(np.uint8)
                X_processed[i] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            X_processed = X.copy()
        
        # 归一化为[0, 255]范围的uint8类型
        if X_processed.max() <= 1.0:
            X_processed = (X_processed * 255).astype(np.uint8)
        else:
            X_processed = X_processed.astype(np.uint8)
            
        # 根据选择的特征提取方法处理
        if self.feature_method == 'pixel':
            # 直接使用像素值作为特征
            features = X_processed.reshape(X_processed.shape[0], -1)
        elif self.feature_method == 'hog':
            features = self._extract_hog_features(X_processed)
        elif self.feature_method == 'lbp':
            features = self._extract_lbp_features(X_processed)
        elif self.feature_method == 'hog+lbp':
            # 组合HOG和LBP特征
            hog_features = self._extract_hog_features(X_processed)
            lbp_features = self._extract_lbp_features(X_processed)
            features = np.hstack((hog_features, lbp_features))
        else:
            print(f"未知的特征提取方法 '{self.feature_method}', 使用原始像素作为特征")
            features = X_processed.reshape(X_processed.shape[0], -1)
        
        # 缓存结果
        self.feature_cache[cache_key] = features
        
        return features
    
    def _extract_hog_features(self, X: np.ndarray) -> np.ndarray:
        """提取HOG特征"""
        features = []
        
        # 创建HOG描述符
        hog = cv2.HOGDescriptor(
            self.hog_params['winSize'],
            self.hog_params['blockSize'],
            self.hog_params['blockStride'],
            self.hog_params['cellSize'],
            self.hog_params['nbins']
        )
        
        for i in range(X.shape[0]):
            # 确保图像大小正确
            if X[i].shape != self.hog_params['winSize']:
                img = cv2.resize(X[i], self.hog_params['winSize'])
            else:
                img = X[i].copy()
            
            # 提取HOG特征
            feature = hog.compute(img)
            features.append(feature.flatten())
        
        return np.array(features)
    
    def _extract_lbp_features(self, X: np.ndarray) -> np.ndarray:
        """提取LBP特征"""
        from skimage.feature import local_binary_pattern
        
        features = []
        
        for i in range(X.shape[0]):
            img = X[i]
            
            # 计算LBP
            lbp = local_binary_pattern(
                img, 
                self.lbp_params['n_points'], 
                self.lbp_params['radius'], 
                method='uniform'
            )
            
            # 将图像分成网格并计算每个网格的直方图
            grid_x, grid_y = self.lbp_params['grid_x'], self.lbp_params['grid_y']
            height, width = lbp.shape
            h_step, w_step = height // grid_y, width // grid_x
            
            hist_features = []
            for y in range(grid_y):
                for x in range(grid_x):
                    # 提取网格
                    cell = lbp[y*h_step:(y+1)*h_step, x*w_step:(x+1)*w_step]
                    # 计算直方图
                    n_bins = self.lbp_params['n_points'] + 2  # uniform LBP的bins数
                    hist, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins), density=True)
                    hist_features.extend(hist)
            
            features.append(hist_features)
        
        return np.array(features)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        训练SVM模型
        
        Args:
            X_train: 训练图像
            y_train: 训练标签
            
        Returns:
            包含训练结果的字典
        """
        print("开始训练增强版SVM模型...")
        print(f"特征提取方法: {self.feature_method}")
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # 提取特征
        X_features = self._extract_features(X_train)
        print(f"提取的特征形状: {X_features.shape}")
        
        # 确定PCA组件数量
        n_components = min(self.n_components, X_features.shape[0] - 1, X_features.shape[1])
        print(f"使用PCA降维，保留{n_components}个主成分")
        
        if self.use_ensemble:
            # 使用集成学习
            self._train_ensemble(X_features, y_encoded, n_components)
        else:
            # 使用单一SVM
            self._train_single_model(X_features, y_encoded, n_components)
        
        self.is_trained = True
        
        # 返回训练结果
        result = {
            'model_type': 'SVM_Ensemble' if self.use_ensemble else 'SVM',
            'feature_method': self.feature_method,
            'n_components': n_components,
            'n_classes': len(self.label_encoder.classes_)
        }
        
        if self.best_params:
            result['best_params'] = self.best_params
            
        return result
    
    def _train_single_model(self, X_features: np.ndarray, y_encoded: np.ndarray, n_components: int):
        """训练单一SVM模型"""
        # 创建处理管道
        self.pca = PCA(n_components=n_components, whiten=True)
        self.svm = SVC(
            kernel=self.kernel, 
            C=self.C, 
            gamma=self.gamma, 
            probability=True,
            class_weight=self.class_weight
        )
        
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('pca', self.pca),
            ('svm', self.svm)
        ])
        
        # 执行网格搜索以找到最佳参数
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1]
            }
            
            # 使用5折交叉验证
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=cv, 
                scoring='accuracy', verbose=1
            )
            
            print("执行网格搜索以优化SVM参数...")
            grid_search.fit(X_features, y_encoded)
            
            # 获取最佳参数
            self.best_params = grid_search.best_params_
            print(f"最佳参数: {self.best_params}")
            print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
            
            # 使用最佳参数
            self.pipeline = grid_search.best_estimator_
    
    def _train_ensemble(self, X_features: np.ndarray, y_encoded: np.ndarray, n_components: int):
        """训练集成SVM模型"""
        # 创建多个SVM分类器
        svm_models = [
            ('linear', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components, whiten=True)),
                ('svm', SVC(kernel='linear', probability=True, class_weight=self.class_weight))
            ])),
            ('rbf', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components, whiten=True)),
                ('svm', SVC(kernel='rbf', probability=True, class_weight=self.class_weight))
            ])),
            ('poly', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components, whiten=True)),
                ('svm', SVC(kernel='poly', degree=2, probability=True, class_weight=self.class_weight))
            ]))
        ]
        
        # 创建投票分类器
        self.ensemble = VotingClassifier(
            estimators=svm_models,
            voting='soft'  # 使用概率加权投票
        )
        
        # 训练集成模型
        self.ensemble.fit(X_features, y_encoded)
        print("集成SVM模型训练完成")
        
        # 训练一个基础流水线用于预处理
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, whiten=True)),
        ])
        self.pipeline.fit(X_features, y_encoded)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测人脸标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 提取特征
        X_features = self._extract_features(X)
        
        # 预测
        if self.use_ensemble and self.ensemble is not None:
            y_pred_encoded = self.ensemble.predict(X_features)
        else:
            y_pred_encoded = self.pipeline.predict(X_features)
        
        # 转换回原始标签
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 提取特征
        X_features = self._extract_features(X)
        
        # 预测概率
        if self.use_ensemble and self.ensemble is not None:
            return self.ensemble.predict_proba(X_features)
        else:
            return self.pipeline.predict_proba(X_features)
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法保存")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            # 清除特征缓存，避免保存不必要的大数据
            self.feature_cache = {}
            
            # 保存模型
            save_dict = {
                'pipeline': self.pipeline,
                'ensemble': self.ensemble,
                'label_encoder': self.label_encoder,
                'feature_method': self.feature_method,
                'use_ensemble': self.use_ensemble,
                'is_trained': self.is_trained,
                'best_params': self.best_params,
                'hog_params': self.hog_params,
                'lbp_params': self.lbp_params,
                'class_weight': self.class_weight
            }
            
            # 使用协议4，确保兼容性
            joblib.dump(save_dict, filepath, protocol=4)
            print(f"SVM模型已保存到: {filepath}")
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            # 尝试使用备用保存方式
            try:
                # 单独保存各个组件
                joblib.dump(self.pipeline, filepath + "_pipeline", protocol=4)
                if self.ensemble:
                    joblib.dump(self.ensemble, filepath + "_ensemble", protocol=4)
                joblib.dump(self.label_encoder, filepath + "_encoder", protocol=4)
                
                # 保存配置
                config = {
                    'feature_method': self.feature_method,
                    'use_ensemble': self.use_ensemble,
                    'is_trained': self.is_trained,
                    'best_params': self.best_params,
                    'hog_params': self.hog_params,
                    'lbp_params': self.lbp_params,
                    'class_weight': self.class_weight
                }
                joblib.dump(config, filepath + "_config", protocol=4)
                print(f"模型已通过备用方式保存")
            except Exception as e2:
                print(f"备用保存方法也失败: {str(e2)}")
                raise
    
    def load_model(self, filepath: str):
        """加载模型"""
        # 清除特征缓存
        self.feature_cache = {}
        
        if not os.path.exists(filepath):
            # 检查是否存在备用保存文件
            if os.path.exists(filepath + "_config"):
                try:
                    # 加载配置
                    config = joblib.load(filepath + "_config")
                    for key, value in config.items():
                        setattr(self, key, value)
                    
                    # 加载管道
                    if os.path.exists(filepath + "_pipeline"):
                        self.pipeline = joblib.load(filepath + "_pipeline")
                    
                    # 加载集成
                    if os.path.exists(filepath + "_ensemble"):
                        self.ensemble = joblib.load(filepath + "_ensemble")
                    
                    # 加载编码器
                    if os.path.exists(filepath + "_encoder"):
                        self.label_encoder = joblib.load(filepath + "_encoder")
                    
                    print(f"SVM模型已从备用文件加载")
                    return
                except Exception as e:
                    print(f"加载备用文件失败: {str(e)}")
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        try:
            # 加载模型
            model_data = joblib.load(filepath)
            
            # 恢复模型状态
            self.pipeline = model_data.get('pipeline')
            self.ensemble = model_data.get('ensemble')
            self.label_encoder = model_data.get('label_encoder')
            self.feature_method = model_data.get('feature_method', 'pixel')
            self.use_ensemble = model_data.get('use_ensemble', False)
            self.is_trained = model_data.get('is_trained', False)
            self.best_params = model_data.get('best_params')
            self.hog_params = model_data.get('hog_params', self.hog_params)
            self.lbp_params = model_data.get('lbp_params', self.lbp_params)
            self.class_weight = model_data.get('class_weight', 'balanced')
            
            print(f"SVM模型已从 {filepath} 加载")
            print(f"特征提取方法: {self.feature_method}")
            print(f"类别数: {len(self.label_encoder.classes_)}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            X_test: 测试图像
            y_test: 真实标签
            
        Returns:
            包含评估指标的字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 预测
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)
        
        # 各类别的准确率
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        class_accuracy_dict = dict(zip(self.label_encoder.classes_, class_accuracy))
        
        print(f"总体准确率: {accuracy:.4f}")
        print("各类别准确率:")
        for cls, acc in class_accuracy_dict.items():
            print(f"  {cls}: {acc:.4f}")
        
        return {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy_dict,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def visualize_features(self, X: np.ndarray, y: np.ndarray, n_samples: int = 5):
        """
        可视化提取的特征
        
        Args:
            X: 输入图像
            y: 图像标签
            n_samples: 每个类别显示的样本数
        """
        # 获取类别列表
        classes = np.unique(y)
        n_classes = len(classes)
        
        # 为每个类别选择样本
        selected_indices = []
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) >= n_samples:
                selected_indices.extend(cls_indices[:n_samples])
            else:
                selected_indices.extend(cls_indices)
        
        selected_X = X[selected_indices]
        selected_y = y[selected_indices]
        
        # 提取特征
        features = self._extract_features(selected_X)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 使用PCA或t-SNE降维
        from sklearn.manifold import TSNE
        
        # t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_scaled)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        # 为每个类别使用不同颜色
        unique_classes = np.unique(selected_y)
        cmap = plt.cm.get_cmap('tab10', len(unique_classes))
        
        for i, cls in enumerate(unique_classes):
            idx = np.where(selected_y == cls)[0]
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                      color=cmap(i), label=str(cls), alpha=0.7)
        
        plt.title(f'特征可视化 (使用{self.feature_method}特征)')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        绘制混淆矩阵
        
        Args:
            X_test: 测试图像
            y_test: 真实标签
        """
        import seaborn as sns
        
        # 评估模型
        eval_results = self.evaluate(X_test, y_test)
        cm = eval_results['confusion_matrix']
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=self.label_encoder.classes_, 
                   yticklabels=self.label_encoder.classes_)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.show()

    def debug_prediction(self, X_sample: np.ndarray, y_true=None, top_k=3):
        """
        调试单个样本的预测过程
        
        Args:
            X_sample: 单个图像样本
            y_true: 可选的真实标签
            top_k: 显示前k个预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 确保输入是单个样本
        if X_sample.ndim == 4:  # 批量输入，取第一个
            X_sample = X_sample[0:1]
        elif X_sample.ndim == 2:  # 单个灰度图，增加维度
            X_sample = X_sample.reshape(1, *X_sample.shape)
        
        # 提取特征
        X_features = self._extract_features(X_sample)
        
        # 获取预测结果和概率
        if self.use_ensemble and self.ensemble is not None:
            y_proba = self.ensemble.predict_proba(X_features)[0]
        else:
            y_proba = self.pipeline.predict_proba(X_features)[0]
        
        # 获取前K个预测
        top_k_indices = np.argsort(y_proba)[::-1][:top_k]
        top_k_proba = y_proba[top_k_indices]
        top_k_labels = self.label_encoder.inverse_transform(top_k_indices)
        
        # 绘制样本和预测结果
        plt.figure(figsize=(12, 5))
        
        # 显示样本图像
        plt.subplot(1, 2, 1)
        if X_sample.ndim == 4 and X_sample.shape[3] == 1:
            plt.imshow(X_sample[0, :, :, 0], cmap='gray')
        elif X_sample.ndim == 4 and X_sample.shape[3] == 3:
            plt.imshow(X_sample[0])
        else:
            plt.imshow(X_sample[0], cmap='gray')
        
        if y_true is not None:
            plt.title(f"真实标签: {y_true}")
        else:
            plt.title("输入样本")
        plt.axis('off')
        
        # 显示预测结果
        plt.subplot(1, 2, 2)
        colors = ['green' if y_true == label else 'red' for label in top_k_labels] if y_true is not None else ['blue'] * top_k
        
        plt.barh(range(top_k), top_k_proba, color=colors)
        plt.yticks(range(top_k), top_k_labels)
        plt.xlabel('预测概率')
        plt.title('模型预测结果')
        
        # 添加概率值标签
        for i, v in enumerate(top_k_proba):
            plt.text(v + 0.01, i, f"{v:.4f}")
        
        plt.tight_layout()
        plt.show()
        
        # 返回预测结果
        return {
            'top_labels': top_k_labels,
            'top_probabilities': top_k_proba
        }

class AdvancedSVMFaceRecognitionModel(SVMFaceRecognitionModel):
    """
    扩展版SVM人脸识别模型，具有更多高级功能
    """
    
    def __init__(self, **kwargs):
        """
        初始化高级SVM模型，转发所有参数到父类
        """
        super().__init__(**kwargs)
        # 扩展功能可以在这里添加