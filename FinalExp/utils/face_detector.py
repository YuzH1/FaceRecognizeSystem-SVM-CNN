import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Union

class FaceDetector:
    """
    LFW数据集优化的人脸检测器
    - 针对LFW数据集特点进行参数优化
    - 提供一致的预处理流程
    - 确保训练和预测使用相同的数据格式
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        初始化LFW人脸检测器
        
        Args:
            use_gpu: 是否使用GPU加速检测(如果可用)
        """
        # 状态变量
        self.initialized = False
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        # 检测器配置
        self.detect_mode = 'cascade'  # 'cascade', 'dnn'
        self.preprocessing = 'equalize'  # 'equalize', 'clahe', 'normalize', 'none'
        
        # 人脸检测器
        self.face_cascade = None
        self.dnn_face_detector = None
        
        # 初始化检测器
        self.initialize()
        
    def initialize(self):
        """初始化检测器"""
        print("初始化LFW优化的人脸检测器...")
        
        # 1. 加载Haar级联分类器
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if not self.face_cascade.empty():
                print("✅ Haar级联分类器加载成功")
            else:
                print("❌ Haar级联分类器为空")
                self.face_cascade = None
        else:
            print("❌ Haar级联分类器文件不存在")
        
        # 2. 尝试加载DNN检测器
        try:
            # 先尝试从OpenCV提供的路径加载模型
            model_path = os.path.join(cv2.data.haarcascades, "..", "face_detector", "opencv_face_detector_uint8.pb")
            config_path = os.path.join(cv2.data.haarcascades, "..", "face_detector", "opencv_face_detector.pbtxt")
            
            # 如果OpenCV安装包中没有模型文件，尝试从本地加载
            if not (os.path.exists(model_path) and os.path.exists(config_path)):
                model_path = "models/opencv_face_detector_uint8.pb"
                config_path = "MachineLearning\FinalExp\models\opencv_face_detector.pbtxt"
                
                # 如果本地也没有，跳过DNN模型
                if not (os.path.exists(model_path) and os.path.exists(config_path)):
                    print("❌ DNN模型文件不存在，跳过DNN检测功能")
                    self.dnn_face_detector = None
                    self.detect_mode = 'cascade'  # 默认使用级联分类器
                    self.initialized = self.face_cascade is not None
                    return
            
            # 加载模型
            self.dnn_face_detector = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            
            # GPU加速配置
            if self.use_gpu:
                self.dnn_face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.dnn_face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("✅ DNN人脸检测模型加载成功 (GPU加速)")
            else:
                print("✅ DNN人脸检测模型加载成功")
        except Exception as e:
            print(f"❌ DNN模型加载失败: {str(e)}")
            self.dnn_face_detector = None
        
        # 设置检测模式优先级
        if self.dnn_face_detector is not None:
            self.detect_mode = 'dnn'  # 如果DNN可用，优先使用
        elif self.face_cascade is not None:
            self.detect_mode = 'cascade'  # 否则使用Haar级联
        else:
            print("警告: 所有检测器加载失败")
            return
        
        print(f"初始化完成，当前检测模式: {self.detect_mode}")
        self.initialized = True
    
    
    
    def detect_faces(self, image: np.ndarray, 
                   min_face_size: Tuple[int, int] = (30, 30),
                   confidence_threshold: float = 0.7) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像
            min_face_size: 最小人脸尺寸(宽,高)
            confidence_threshold: 检测置信度阈值
            
        Returns:
            人脸位置列表，每个元素为(x, y, w, h)
        """
        if not self.initialized:
            print("错误: 检测器未成功初始化")
            return []
        
        if image is None or image.size == 0:
            print("错误: 输入图像为空")
            return []
        
        # 预处理图像
        processed_image = self._preprocess_for_detection(image)
        
        # 根据当前模式选择检测方法
        if self.detect_mode == 'dnn' and self.dnn_face_detector is not None:
            faces = self._detect_with_dnn(processed_image, confidence_threshold)
        elif self.detect_mode == 'cascade' and self.face_cascade is not None:
            faces = self._detect_with_cascade(processed_image, min_face_size)
        else:
            print("错误: 无可用的检测方法")
            return []
        
        # 不进行额外的后处理，保持简单可靠
        if faces:
            print(f"检测到 {len(faces)} 个人脸")
        else:
            print("未检测到人脸")
            
        return faces
    
    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """图像检测前预处理"""
        # 确保图像有正确的数据类型
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 转换为RGB格式(如果是灰度图)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        return image
    
    def _detect_with_cascade(self, image: np.ndarray, min_face_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """使用级联分类器检测人脸"""
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 为LFW数据集优化的参数
        # LFW图像大多是正面清晰人脸，可以提高scaleFactor以加速检测
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # LFW图像多为标准大小，可用较小的缩放因子
            minNeighbors=5,       # 较高的minNeighbors提高准确性
            minSize=min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 如果检测失败，尝试更宽松的参数
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(min_face_size[0]//2, min_face_size[1]//2),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
        return faces.tolist() if len(faces) > 0 else []
    
    def _detect_with_dnn(self, image: np.ndarray, confidence_threshold: float) -> List[Tuple[int, int, int, int]]:
        """使用DNN模型检测人脸"""
        try:
            (h, w) = image.shape[:2]
            
            # 创建blob
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            
            # 运行模型
            self.dnn_face_detector.setInput(blob)
            detections = self.dnn_face_detector.forward()
            
            # 处理结果
            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # 只保留高置信度的检测结果，提高准确性
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x_end, y_end) = box.astype("int")
                    
                    # 确保坐标在图像范围内
                    x = max(0, x)
                    y = max(0, y)
                    x_end = min(w, x_end)
                    y_end = min(h, y_end)
                    
                    # 计算宽度和高度
                    width = x_end - x
                    height = y_end - y
                    
                    if width > 0 and height > 0:
                        faces.append((x, y, width, height))
            
            return faces
            
        except Exception as e:
            print(f"DNN检测失败: {str(e)}")
            return []
    
    # 在LFWFaceDetector类中修改extract_face方法
    def extract_face(self, image: np.ndarray, 
                    target_size: Tuple[int, int] = (64, 64),
                    padding_ratio: float = 0.0,
                    always_return: bool = False,
                    size: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """
        提取人脸区域并进行标准化处理
        
        Args:
            image: 输入图像
            target_size: 输出图像大小
            padding_ratio: 周围区域填充比例
            always_return: 若为True，未检测到人脸时返回原图缩放版本
            size: 兼容旧API的参数(等同于target_size)
            
        Returns:
            处理后的人脸图像，未检测到则返回None
        """
        # 处理兼容性参数
        if size is not None:
            target_size = size
        
        if image is None or image.size == 0:
            return None
        
        # 检测人脸
        faces = self.detect_faces(image)
        
        # 未检测到人脸
        if not faces:
            if always_return:
                # 将整个图像视为人脸
                print("未检测到人脸，使用整图缩放")
                return self._process_face_region(image, target_size)
            else:
                print("未检测到人脸，返回None")
                return None
        
        # 选择最大的人脸(大多数LFW图像只有一个主要人脸)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # 添加边缘填充
        if padding_ratio > 0:
            padding_x = int(w * padding_ratio)
            padding_y = int(h * padding_ratio)
            
            # 边界检查
            x_start = max(0, x - padding_x)
            y_start = max(0, y - padding_y)
            x_end = min(image.shape[1], x + w + padding_x)
            y_end = min(image.shape[0], y + h + padding_y)
        else:
            x_start, y_start = x, y
            x_end, y_end = x + w, y + h
        
        # 提取并处理人脸区域
        face_region = image[y_start:y_end, x_start:x_end]
        return self._process_face_region(face_region, target_size)
    
    def _process_face_region(self, face_region: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """处理提取的人脸区域"""
        if face_region.size == 0:
            return None
        
        # 转换为灰度图(如果不是)
        if len(face_region.shape) == 3:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_region
        
        # 调整大小
        face_resized = cv2.resize(face_gray, target_size)
        
        # 应用预处理方法
        if self.preprocessing == 'equalize':
            # 直方图均衡化 - LFW数据集常用的预处理方法
            face_processed = cv2.equalizeHist(face_resized)
        elif self.preprocessing == 'clahe':
            # CLAHE可以在保留局部对比度的同时限制噪声放大
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_processed = clahe.apply(face_resized)
        elif self.preprocessing == 'normalize':
            # Min-Max归一化
            face_processed = cv2.normalize(face_resized, None, 0, 255, cv2.NORM_MINMAX)
        else:
            # 不做额外处理
            face_processed = face_resized
        
        # 确保数据类型正确并归一化到[0,1]范围
        return face_processed.astype(np.float32) / 255.0
    
    def set_preprocessing(self, method: str):
        """
        设置预处理方法
        
        Args:
            method: 预处理方法 ('equalize', 'clahe', 'normalize', 'none')
        """
        valid_methods = ['equalize', 'clahe', 'normalize', 'none']
        if method not in valid_methods:
            print(f"无效的预处理方法: {method}")
            print(f"有效选项: {valid_methods}")
            return
        
        self.preprocessing = method
        print(f"预处理方法已设置为: {method}")
    
    def set_detect_mode(self, mode: str):
        """
        设置人脸检测模式
        
        Args:
            mode: 检测模式 ('dnn', 'cascade', 'auto')
        """
        if mode == 'dnn' and self.dnn_face_detector is None:
            print("DNN模型未加载，无法设置为DNN模式")
            return
            
        if mode == 'cascade' and self.face_cascade is None:
            print("级联分类器未加载，无法设置为级联模式")
            return
            
        if mode == 'auto':
            # 自动选择最佳可用方法
            if self.dnn_face_detector is not None:
                self.detect_mode = 'dnn'
            elif self.face_cascade is not None:
                self.detect_mode = 'cascade'
            else:
                print("没有可用的检测方法")
                return
        else:
            self.detect_mode = mode
            
        print(f"检测模式已设置为: {self.detect_mode}")
    
    def draw_faces(self, image: np.ndarray, 
                  faces: Optional[List[Tuple[int, int, int, int]]] = None,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制检测到的人脸
        
        Args:
            image: 输入图像
            faces: 人脸位置列表，如为None则自动检测
            color: 框的颜色
            thickness: 线条粗细
            
        Returns:
            绘制了人脸框的图像
        """
        if image is None:
            return None
        
        # 复制原图
        result = image.copy()
        
        # 如果未提供人脸位置，自动检测
        if faces is None:
            faces = self.detect_faces(image)
        
        # 绘制人脸框
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        
        return result

    def visualize_preprocessing(self, image: np.ndarray) -> None:
        """可视化不同预处理方法的效果"""
        if image is None:
            return
        
        # 检测人脸
        faces = self.detect_faces(image)
        
        if not faces:
            print("未检测到人脸，无法可视化")
            return
        
        # 提取最大的人脸
        x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
        face_region = image[y:y+h, x:x+w]
        
        # 转为灰度图
        if len(face_region.shape) == 3:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_region
        
        # 调整大小
        face_resized = cv2.resize(face_gray, (64, 64))
        
        # 应用不同的预处理方法
        face_original = face_resized
        face_equalized = cv2.equalizeHist(face_resized)
        face_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face_resized)
        face_normalized = cv2.normalize(face_resized, None, 0, 255, cv2.NORM_MINMAX)
        
        # 可视化
        plt.figure(figsize=(12, 3))
        
        plt.subplot(141)
        plt.title('原始')
        plt.imshow(face_original, cmap='gray')
        plt.axis('off')
        
        plt.subplot(142)
        plt.title('直方图均衡化')
        plt.imshow(face_equalized, cmap='gray')
        plt.axis('off')
        
        plt.subplot(143)
        plt.title('CLAHE')
        plt.imshow(face_clahe, cmap='gray')
        plt.axis('off')
        
        plt.subplot(144)
        plt.title('归一化')
        plt.imshow(face_normalized, cmap='gray')
        plt.axis('off')
        
        plt.suptitle('LFW人脸预处理方法对比')
        plt.tight_layout()
        plt.show()
    
    def process_lfw_dataset(self, lfw_dir: str, output_dir: str, target_size: Tuple[int, int] = (64, 64)) -> Dict:
        """
        处理整个LFW数据集
        
        Args:
            lfw_dir: LFW数据集目录
            output_dir: 输出目录
            target_size: 处理后的图像大小
            
        Returns:
            处理结果统计
        """
        if not os.path.exists(lfw_dir):
            print(f"LFW数据集目录不存在: {lfw_dir}")
            return {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'persons': 0
        }
        
        # 遍历所有人物文件夹
        print(f"处理LFW数据集: {lfw_dir}")
        
        for person_name in sorted(os.listdir(lfw_dir)):
            person_dir = os.path.join(lfw_dir, person_name)
            
            # 检查是否为目录
            if not os.path.isdir(person_dir):
                continue
                
            # 创建输出子目录
            person_output_dir = os.path.join(output_dir, person_name)
            os.makedirs(person_output_dir, exist_ok=True)
            
            # 统计数据
            stats['persons'] += 1
            processed_count = 0
            failed_count = 0
            
            # 处理此人物的所有图像
            image_files = [f for f in os.listdir(person_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                stats['total_images'] += 1
                
                try:
                    # 加载图像
                    img_path = os.path.join(person_dir, img_file)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        print(f"无法读取图像: {img_path}")
                        failed_count += 1
                        continue
                    
                    # 提取人脸
                    face = self.extract_face(
                        image, 
                        target_size=target_size,
                        padding_ratio=0.1,
                        always_return=True  # LFW应该总是有人脸
                    )
                    
                    if face is None:
                        print(f"无法提取人脸: {img_path}")
                        failed_count += 1
                        continue
                    
                    # 保存处理后的图像
                    output_path = os.path.join(person_output_dir, img_file)
                    face_img = (face * 255).astype(np.uint8)
                    cv2.imwrite(output_path, face_img)
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"处理图像时出错 {img_path}: {str(e)}")
                    failed_count += 1
            
            stats['processed_images'] += processed_count
            stats['failed_images'] += failed_count
            
            print(f"处理完成: {person_name} - 成功: {processed_count}, 失败: {failed_count}")
        
        # 打印总结
        print("\n=== LFW处理完成 ===")
        print(f"总人数: {stats['persons']}")
        print(f"总图像: {stats['total_images']}")
        print(f"成功处理: {stats['processed_images']}")
        print(f"处理失败: {stats['failed_images']}")
        print(f"成功率: {stats['processed_images']/stats['total_images']*100:.1f}%")
        
        return stats


# 用于演示和测试的函数
def lfw_detector_demo():
    """演示LFW人脸检测器的功能"""
    # 创建检测器
    detector = FaceDetector()
    
    # 创建示例图像进行测试
    image = create_test_face_image()
    
    # 测试不同的预处理方法
    print("\n测试不同的预处理方法:")
    for method in ['equalize', 'clahe', 'normalize', 'none']:
        detector.set_preprocessing(method)
        face = detector.extract_face(image)
        if face is not None:
            plt.figure(figsize=(4, 4))
            plt.imshow(face, cmap='gray')
            plt.title(f"预处理: {method}")
            plt.axis('off')
            plt.show()
    
    # 可视化人脸检测结果
    faces = detector.detect_faces(image)
    result = detector.draw_faces(image, faces)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("原始图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"检测结果: {len(faces)} 个人脸")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_test_face_image():
    """创建测试用的人脸图像"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # 设置背景色
    img.fill(50)
    
    # 画人脸轮廓
    cv2.ellipse(img, (150, 150), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    
    # 画眼睛
    cv2.circle(img, (120, 130), 8, (50, 50, 50), -1)  # 左眼
    cv2.circle(img, (180, 130), 8, (50, 50, 50), -1)  # 右眼
    cv2.circle(img, (123, 127), 3, (255, 255, 255), -1)  # 左眼高光
    cv2.circle(img, (183, 127), 3, (255, 255, 255), -1)  # 右眼高光
    
    # 画眉毛
    cv2.ellipse(img, (120, 115), (15, 5), 0, 0, 180, (100, 80, 60), 2)
    cv2.ellipse(img, (180, 115), (15, 5), 0, 0, 180, (100, 80, 60), 2)
    
    # 画鼻子
    cv2.line(img, (150, 140), (150, 170), (150, 130, 110), 2)
    cv2.circle(img, (145, 170), 2, (120, 100, 80), -1)
    cv2.circle(img, (155, 170), 2, (120, 100, 80), -1)
    
    # 画嘴巴
    cv2.ellipse(img, (150, 190), (20, 8), 0, 0, 180, (150, 100, 100), 2)
    
    # 添加一些噪声使其更真实
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # 轻微模糊
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    return img


if __name__ == "__main__":
    lfw_detector_demo()