# utils/data_loader.py
import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
import matplotlib.pyplot as plt
import warnings
from collections import Counter

class FaceDataLoader:
    """人脸数据加载器 - 支持内置数据集和自定义数据集"""
    
    def __init__(self, data_source: str = "olivetti", 
                 data_dir: Optional[str] = None, 
                 image_size: Tuple[int, int] = (64, 64)):
        """
        初始化数据加载器
        
        Args:
            data_source: 数据源类型 ("olivetti", "lfw", "custom", "sample")
            data_dir: 自定义数据集目录（仅当data_source="custom"时使用）
            image_size: 图像尺寸
        """
        self.data_source = data_source
        self.data_dir = data_dir
        self.image_size = image_size
        
        # 验证数据源
        valid_sources = ["olivetti", "lfw", "custom", "sample"]
        if data_source not in valid_sources:
            raise ValueError(f"数据源必须是以下之一: {valid_sources}")
        
        if data_source == "custom" and data_dir is None:
            raise ValueError("使用自定义数据集时必须提供data_dir")
    
    def load_dataset(self, test_size: float = 0.2, 
                    min_faces_per_person: int = 15,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        加载数据集
        
        Args:
            test_size: 测试集比例
            min_faces_per_person: 每人最少需要的人脸数量（仅用于LFW数据集）
            random_state: 随机种子
            
        Returns:
            X_train, X_test, y_train, y_test, label_names
        """
        print(f"开始加载 {self.data_source} 数据集...")
        
        try:
            if self.data_source == "olivetti":
                return self._load_olivetti_dataset(test_size, random_state)
            elif self.data_source == "lfw":
                return self._load_lfw_dataset(test_size, min_faces_per_person, random_state)
            elif self.data_source == "custom":
                return self._load_custom_dataset(test_size, random_state)
            elif self.data_source == "sample":
                return self._load_sample_dataset(test_size, random_state)
            else:
                raise ValueError(f"不支持的数据源: {self.data_source}")
        
        except Exception as e:
            print(f"加载 {self.data_source} 数据集失败: {str(e)}")
            
            # 提供备用方案
            if self.data_source in ["lfw", "custom"]:
                print("尝试加载示例数据集作为备用...")
                return self._load_sample_dataset(test_size, random_state)
            else:
                raise e
    
    def _load_olivetti_dataset(self, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """加载Olivetti faces数据集"""
        print("正在加载Olivetti faces数据集...")
        
        try:
            # 获取数据集
            dataset = fetch_olivetti_faces(shuffle=True, random_state=random_state)
            X = dataset.data  # 形状: (400, 4096) - 400张64x64的图像
            y = dataset.target  # 形状: (400,) - 40个人，每人10张图片
            
            print(f"Olivetti数据集信息:")
            print(f"  总样本数: {X.shape[0]}")
            print(f"  原始图像尺寸: 64x64")
            print(f"  人数: {len(np.unique(y))}")
            print(f"  每人图片数: {X.shape[0] // len(np.unique(y))}")
            
            # 重塑为图像格式
            X_reshaped = X.reshape(-1, 64, 64)
            
            # 调整图像大小（如果需要）
            if self.image_size != (64, 64):
                print(f"  调整图像尺寸: 64x64 -> {self.image_size}")
                X_resized = np.array([
                    cv2.resize(img, self.image_size) for img in X_reshaped
                ])
            else:
                X_resized = X_reshaped
            
            # 确保数据类型正确
            X_resized = X_resized.astype(np.float32)
            
            # 归一化到0-1范围
            if X_resized.max() > 1.0:
                X_resized = X_resized / 255.0
            
            # 添加通道维度用于CNN
            X_final = np.expand_dims(X_resized, axis=-1)
            
            # 创建标签名称
            label_names = [f"Person_{i:02d}" for i in range(len(np.unique(y)))]
            y_names = np.array([label_names[int(label)] for label in y])
            
            # 检查数据分布
            self._check_data_distribution(y_names, label_names)
            
            # 分割数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_names, test_size=test_size, random_state=random_state, stratify=y_names
            )
            
            print(f"  训练集: {len(X_train)} 样本")
            print(f"  测试集: {len(X_test)} 样本")
            print(f"  图像最终尺寸: {X_final.shape[1:]}")
            
            return X_train, X_test, y_train, y_test, label_names
            
        except Exception as e:
            print(f"Olivetti数据集加载失败: {e}")
            raise
    
    def _load_lfw_dataset(self, test_size: float, min_faces_per_person: int, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """加载LFW (Labeled Faces in the Wild)数据集"""
        print("正在加载LFW数据集...")
        
        try:
            # 尝试不同的参数组合以兼容不同版本的sklearn
            fetch_params = {
                'min_faces_per_person': min_faces_per_person,
                'resize': 0.4,  # 调整大小以减少内存使用
                'download_if_missing': True
            }
            
            # 检查sklearn版本并添加相应参数
            try:
                import sklearn
                sklearn_version = sklearn.__version__
                print(f"  检测到sklearn版本: {sklearn_version}")
                
                # 新版本sklearn支持random_state参数
                if hasattr(fetch_lfw_people, '__code__') and 'random_state' in fetch_lfw_people.__code__.co_varnames:
                    fetch_params['random_state'] = random_state
                    
            except:
                pass
            
            # 获取数据集
            print(f"  获取参数: {fetch_params}")
            dataset = fetch_lfw_people(**fetch_params)
            
            X = dataset.data
            y = dataset.target
            label_names = dataset.target_names
            
            print(f"LFW数据集信息:")
            print(f"  总样本数: {X.shape[0]}")
            print(f"  原始图像尺寸: {dataset.images.shape[1:]} -> 重塑为 {int(np.sqrt(X.shape[1]))}x{int(np.sqrt(X.shape[1]))}")
            print(f"  人数: {len(label_names)}")
            print(f"  最少每人图片数: {min_faces_per_person}")
            
            # 检查是否有足够的数据
            if len(X) == 0:
                raise ValueError("LFW数据集为空，可能是min_faces_per_person设置过高")
            
            # 重塑为图像格式
            img_h, img_w = dataset.images.shape[1], dataset.images.shape[2]
            X_reshaped = X.reshape(-1, img_h, img_w)
            
            # 调整图像大小
            print(f"  调整图像尺寸: {img_h}x{img_w} -> {self.image_size}")
            X_resized = np.array([
                cv2.resize(img, self.image_size) for img in X_reshaped
            ])
            
            # 确保数据类型正确
            X_resized = X_resized.astype(np.float32)
            
            # 归一化到0-1范围
            if X_resized.max() > 1.0:
                X_resized = X_resized / 255.0
            
            # 添加通道维度
            X_final = np.expand_dims(X_resized, axis=-1)
            
            # 转换标签为字符串
            y_names = np.array([label_names[int(label)] for label in y])
            
            # 检查数据分布
            self._check_data_distribution(y_names, label_names)
            
            # 分割数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_names, test_size=test_size, random_state=random_state, stratify=y_names
            )
            
            print(f"  训练集: {len(X_train)} 样本")
            print(f"  测试集: {len(X_test)} 样本")
            print(f"  图像最终尺寸: {X_final.shape[1:]}")
            
            return X_train, X_test, y_train, y_test, list(label_names)
            
        except Exception as e:
            error_message = str(e)
            print(f"加载LFW数据集失败: {error_message}")
            
            # 提供具体的错误建议
            if "random_state" in error_message:
                print("提示: 检测到sklearn版本兼容性问题")
                print("  建议: pip install --upgrade scikit-learn")
            elif "network" in error_message.lower() or "download" in error_message.lower():
                print("提示: LFW数据集需要网络连接下载")
                print("  建议: 检查网络连接或使用其他数据集")
            elif "min_faces_per_person" in error_message:
                print(f"提示: min_faces_per_person={min_faces_per_person} 可能过高")
                print("  建议: 降低 min_faces_per_person 到 10-20")
            
            raise
    
    def _load_custom_dataset(self, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """加载自定义数据集"""
        print(f"从 {self.data_dir} 加载自定义数据集...")
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        
        images = []
        labels = []
        label_names = []
        
        # 支持的图像格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 遍历每个人的文件夹
        for person_name in sorted(os.listdir(self.data_dir)):
            person_dir = os.path.join(self.data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            person_images = []
            print(f"  加载 {person_name} 的图像...")
            
            # 加载该人的所有图像
            for image_file in sorted(os.listdir(person_dir)):
                file_ext = os.path.splitext(image_file)[1].lower()
                if file_ext in valid_extensions:
                    image_path = os.path.join(person_dir, image_file)
                    
                    try:
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        
                        if image is not None:
                            # 调整图像大小
                            image_resized = cv2.resize(image, self.image_size)
                            
                            # 确保数据类型正确
                            image_resized = image_resized.astype(np.float32)
                            
                            # 归一化
                            if image_resized.max() > 1.0:
                                image_resized = image_resized / 255.0
                            
                            person_images.append(image_resized)
                        else:
                            print(f"    警告: 无法加载图像 {image_file}")
                            
                    except Exception as e:
                        print(f"    错误: 加载图像 {image_file} 时出错: {e}")
            
            # 只有当该人有图像时才添加到数据集
            if len(person_images) > 0:
                label_names.append(person_name)
                for img in person_images:
                    images.append(img)
                    labels.append(person_name)
                print(f"    加载了 {len(person_images)} 张图像")
            else:
                print(f"    警告: {person_name} 文件夹中没有有效图像")
        
        if len(images) == 0:
            raise ValueError("未找到任何有效的图像文件")
        
        # 转换为numpy数组
        X = np.array(images)
        y = np.array(labels)
        
        # 为CNN添加通道维度
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        
        print(f"自定义数据集信息:")
        print(f"  总样本数: {len(X)}")
        print(f"  类别数: {len(label_names)}")
        print(f"  图像尺寸: {X.shape[1:]}")
        
        # 检查数据分布
        self._check_data_distribution(y, label_names)
        
        # 检查是否有足够的样本进行分层抽样
        min_samples_per_class = min(Counter(y).values())
        if min_samples_per_class < 2:
            print("  警告: 某些类别样本过少，无法进行分层抽样")
            stratify = None
        else:
            stratify = y
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        return X_train, X_test, y_train, y_test, label_names
    
    def _load_sample_dataset(self, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """加载示例数据集（用于测试和演示）"""
        print("正在生成示例数据集...")
        
        np.random.seed(random_state)
        
        # 生成参数
        n_people = 5
        n_images_per_person = 20
        
        images = []
        labels = []
        label_names = [f"Sample_Person_{i+1}" for i in range(n_people)]
        
        for person_id in range(n_people):
            print(f"  生成 {label_names[person_id]} 的图像...")
            
            for img_id in range(n_images_per_person):
                # 生成合成人脸图像
                img = self._generate_synthetic_face(person_id, img_id)
                images.append(img)
                labels.append(label_names[person_id])
        
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        # 添加通道维度
        X = np.expand_dims(X, axis=-1)
        
        print(f"示例数据集信息:")
        print(f"  总样本数: {len(X)}")
        print(f"  类别数: {len(label_names)}")
        print(f"  每人图片数: {n_images_per_person}")
        print(f"  图像尺寸: {X.shape[1:]}")
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        return X_train, X_test, y_train, y_test, label_names
    
    def _generate_synthetic_face(self, person_id: int, variation_id: int) -> np.ndarray:
        """生成合成人脸图像"""
        # 创建基础图像
        img = np.zeros(self.image_size, dtype=np.uint8)
        
        # 基于person_id和variation_id设置随机种子
        np.random.seed(person_id * 100 + variation_id)
        
        # 图像中心
        center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
        
        # 人脸参数（基于person_id确保一致性，基于variation_id添加变化）
        base_radius = min(self.image_size) // 3
        face_radius = base_radius + np.random.randint(-3, 4)
        
        # 脸部位置微调
        face_center_x = center_x + np.random.randint(-5, 6)
        face_center_y = center_y + np.random.randint(-5, 6)
        
        # 画脸部轮廓
        face_color = 150 + person_id * 20 + np.random.randint(-10, 11)
        cv2.circle(img, (face_center_x, face_center_y), face_radius, face_color, -1)
        
        # 眼睛
        eye_y = face_center_y - face_radius // 3
        left_eye_x = face_center_x - face_radius // 2
        right_eye_x = face_center_x + face_radius // 2
        eye_size = 2 + np.random.randint(0, 3)
        
        cv2.circle(img, (left_eye_x, eye_y), eye_size, 50, -1)
        cv2.circle(img, (right_eye_x, eye_y), eye_size, 50, -1)
        
        # 鼻子
        nose_y = face_center_y
        nose_length = face_radius // 4
        cv2.line(img, (face_center_x, nose_y - nose_length//2), 
                (face_center_x, nose_y + nose_length//2), 100, 1)
        
        # 嘴巴
        mouth_y = face_center_y + face_radius // 3
        mouth_width = face_radius // 3
        mouth_height = face_radius // 8
        cv2.ellipse(img, (face_center_x, mouth_y), (mouth_width, mouth_height), 
                   0, 0, 180, 80, 1)
        
        # 添加一些特征变化（基于person_id）
        if person_id % 2 == 0:
            # 添加眉毛
            cv2.line(img, (left_eye_x - 5, eye_y - 8), (left_eye_x + 5, eye_y - 6), 70, 1)
            cv2.line(img, (right_eye_x - 5, eye_y - 6), (right_eye_x + 5, eye_y - 8), 70, 1)
        
        # 添加噪声和纹理
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # 高斯模糊以使图像更自然
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # 归一化到0-1范围
        return img.astype(np.float32) / 255.0
    
    def _check_data_distribution(self, y: np.ndarray, label_names: List[str]):
        """检查数据分布"""
        counter = Counter(y)
        
        print(f"  数据分布:")
        for label in label_names[:10]:  # 只显示前10个
            count = counter.get(label, 0)
            print(f"    {label}: {count} 样本")
        
        if len(label_names) > 10:
            print(f"    ... 还有 {len(label_names) - 10} 个类别")
        
        # 检查数据不平衡
        counts = list(counter.values())
        if len(counts) > 1:
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > 3:
                print(f"  ⚠️ 数据不平衡严重 (最大/最小比例: {imbalance_ratio:.1f})")
            elif imbalance_ratio > 2:
                print(f"  ⚠️ 轻微数据不平衡 (最大/最小比例: {imbalance_ratio:.1f})")
    
    def visualize_samples(self, X: np.ndarray, y: np.ndarray, 
                         label_names: List[str], n_samples: int = 12,
                         title: str = "数据集样本"):
        """可视化数据集样本"""
        print(f"显示{title}...")
        
        # 确保有足够的样本
        n_samples = min(n_samples, len(X))
        
        # 尝试每个类别选择一些样本
        unique_labels = np.unique(y)
        samples_per_class = max(1, n_samples // len(unique_labels))
        
        selected_indices = []
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            selected = np.random.choice(label_indices, 
                                      min(samples_per_class, len(label_indices)), 
                                      replace=False)
            selected_indices.extend(selected)
        
        # 如果还需要更多样本，随机选择
        while len(selected_indices) < n_samples:
            remaining = list(set(range(len(X))) - set(selected_indices))
            if not remaining:
                break
            selected_indices.append(np.random.choice(remaining))
        
        selected_indices = selected_indices[:n_samples]
        
        # 计算子图布局
        n_cols = 4
        n_rows = (len(selected_indices) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 3 * n_rows))
        plt.suptitle(title, fontsize=16)
        
        for i, idx in enumerate(selected_indices):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # 显示图像
            if len(X[idx].shape) == 3 and X[idx].shape[-1] == 1:
                plt.imshow(X[idx].squeeze(), cmap='gray')
            elif len(X[idx].shape) == 2:
                plt.imshow(X[idx], cmap='gray')
            else:
                plt.imshow(X[idx])
            
            plt.title(f"{y[idx]}", fontsize=10)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_dataset_info(self) -> dict:
        """获取数据集信息"""
        info = {
            'data_source': self.data_source,
            'image_size': self.image_size
        }
        
        if self.data_source == "olivetti":
            info.update({
                'description': 'Olivetti faces数据集 - 40人，每人10张图片',
                'total_samples': 400,
                'num_classes': 40,
                'original_size': (64, 64),
                'advantages': '稳定可靠，无需网络下载',
                'disadvantages': '数据量较小，人物较少'
            })
        elif self.data_source == "lfw":
            info.update({
                'description': 'LFW (Labeled Faces in the Wild)数据集',
                'note': '样本数取决于min_faces_per_person参数',
                'advantages': '真实人脸，数据量大',
                'disadvantages': '需要网络下载，可能有兼容性问题'
            })
        elif self.data_source == "custom":
            info.update({
                'description': f'自定义数据集来自: {self.data_dir}',
                'data_dir': self.data_dir,
                'advantages': '可控数据，适合特定任务',
                'disadvantages': '需要手动准备数据'
            })
        elif self.data_source == "sample":
            info.update({
                'description': '合成示例数据集用于测试',
                'advantages': '快速生成，无依赖',
                'disadvantages': '非真实数据，仅用于测试'
            })
        
        return info


# 创建便捷函数
def load_olivetti_faces(image_size: Tuple[int, int] = (64, 64), 
                       test_size: float = 0.2,
                       random_state: int = 42):
    """便捷函数：加载Olivetti faces数据集"""
    loader = FaceDataLoader("olivetti", image_size=image_size)
    return loader.load_dataset(test_size, random_state=random_state)

def load_lfw_faces(image_size: Tuple[int, int] = (64, 64), 
                  test_size: float = 0.2, 
                  min_faces_per_person: int = 20,
                  random_state: int = 42):
    """便捷函数：加载LFW数据集"""
    loader = FaceDataLoader("lfw", image_size=image_size)
    return loader.load_dataset(test_size, min_faces_per_person, random_state)

def load_custom_faces(data_dir: str, 
                     image_size: Tuple[int, int] = (64, 64), 
                     test_size: float = 0.2,
                     random_state: int = 42):
    """便捷函数：加载自定义数据集"""
    loader = FaceDataLoader("custom", data_dir=data_dir, image_size=image_size)
    return loader.load_dataset(test_size, random_state=random_state)

def load_sample_faces(image_size: Tuple[int, int] = (64, 64), 
                     test_size: float = 0.2,
                     random_state: int = 42):
    """便捷函数：加载示例数据集"""
    loader = FaceDataLoader("sample", image_size=image_size)
    return loader.load_dataset(test_size, random_state=random_state)


# 测试函数
def test_all_datasets():
    """测试所有数据集加载"""
    print("=== 测试所有数据集加载器 ===\n")
    
    datasets_to_test = [
        ("sample", {}),
        ("olivetti", {}),
        # ("lfw", {"min_faces_per_person": 20}),  # 可能需要网络
    ]
    
    for dataset_name, params in datasets_to_test:
        print(f"测试 {dataset_name} 数据集:")
        try:
            loader = FaceDataLoader(dataset_name, image_size=(64, 64))
            X_train, X_test, y_train, y_test, label_names = loader.load_dataset(**params)
            
            print(f"✅ {dataset_name} 数据集加载成功")
            print(f"   训练集: {X_train.shape}")
            print(f"   测试集: {X_test.shape}")
            print(f"   类别数: {len(label_names)}")
            
            # 可视化样本
            loader.visualize_samples(X_train, y_train, label_names, 
                                   n_samples=8, title=f"{dataset_name}数据集样本")
            
        except Exception as e:
            print(f"❌ {dataset_name} 数据集加载失败: {e}")
        
        print("-" * 50)


if __name__ == "__main__":
    test_all_datasets()