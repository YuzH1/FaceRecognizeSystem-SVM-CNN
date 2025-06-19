# 人脸识别系统 (Face Recognition System)

这是一个功能全面的桌面端人脸识别系统，采用 Python 和 Tkinter 构建。系统集成了两种主流的人脸识别技术路线：基于传统特征工程的**支持向量机 (SVM)** 和基于深度学习的**卷积神经网络 (CNN)**。用户可以通过图形化界面轻松完成数据集加载、模型训练、性能评估、模型对比和实时识别等一系列操作。

  
![Uploading image.png…]()


## ✨ 主要功能

*   **双模型支持**: 内置 SVM 和 PyTorch CNN 两种识别模型，方便学习和对比。
*   **模块化架构**: 清晰的项目结构，将界面（GUI）、模型（Models）和工具（Utils）分离，易于维护和扩展。
*   **灵活的数据加载**:
    *   支持加载 `scikit-learn` 内置的 Olivetti Faces 和 LFW 数据集。
    *   支持加载用户自定义的文件夹格式数据集。
    *   提供数据样本可视化功能。
*   **强大的模型实现**:
    *   **SVM**: 集成多种特征提取方法（HOG, LBP, HOG+LBP）、PCA降维和`GridSearchCV`自动超参数优化。
    *   **CNN**: 基于 PyTorch 实现，包含**残差连接 (Residual Blocks)** 和**空间注意力机制 (Spatial Attention)** 的现代化网络架构，并支持数据增强。
*   **全面的模型评估**:
    *   计算准确率、精确率、召回率、F1分数等多种评估指标。
    *   提供详细的分类报告和混淆矩阵可视化。
    *   支持一键对比多个已训练模型的性能。
*   **完整的模型生命周期管理**: 支持模型的训练、保存和加载，方便重复使用。
*   **实时人脸识别**: 支持从外部图片中检测人脸并进行身份预测。
*   **异步处理**: 将耗时的任务（如数据加载和模型训练）置于后台线程执行，保证了GUI界面的流畅性。

## 🛠️ 技术栈

*   **编程语言**: Python 3.x
*   **核心框架**:
    *   **GUI**: `Tkinter` (Python 标准库)
    *   **机器学习**: `scikit-learn`
    *   **深度学习**: `PyTorch`
*   **主要依赖包**:
    *   `OpenCV-Python`: 用于图像处理和人脸检测。
    *   `NumPy`: 用于高效的数值计算。
    *   `Matplotlib` & `Seaborn`: 用于数据和结果的可视化。
    *   `Pillow (PIL)`: 用于在Tkinter中处理和显示图像。
    *   `Pandas`: 用于格式化和展示评估结果。

## 🚀 安装与运行

### 1. 环境准备

建议使用虚拟环境来管理项目依赖。

```bash
# 创建一个虚拟环境 (例如，使用 venv)
python -m venv venv

# 激活虚拟环境
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. 安装依赖

项目的所有依赖项都已列在 `requirements.txt` 文件中。

```bash
# 安装所有必要的包
pip install -r requirements.txt
```

**`requirements.txt` 文件内容:**
```
numpy
opencv-python
scikit-learn
torch
torchvision
matplotlib
Pillow
seaborn
pandas
```

### 3. 运行系统

执行主程序 `main.py` 即可启动图形用户界面。

```bash
python main.py
```

## 📖 使用指南

1.  **加载数据**:
    *   在左上角的“数据管理”面板中，从下拉菜单选择数据集类型（如 `olivetti`）。
    *   点击“加载内置数据集”。
    *   若选择 `custom`，请点击“选择自定义文件夹”并指定您的数据集目录（目录的每个子文件夹代表一个类别）。
    *   加载成功后，右侧会显示数据集信息。

2.  **选择与训练模型**:
    *   在“模型选择”面板中，选择您想使用的模型（SVM 或 PyTorch CNN）。
    *   对于CNN模型，您可以设置学习率、批次大小和训练轮数。
    *   点击“训练模型”按钮。训练过程中的日志会输出在底部的状态栏和右侧的结果区。

3.  **评估模型**:
    *   模型训练完成后，点击“评估模型”按钮。
    *   详细的评估报告（准确率、F1分数等）将显示在右侧结果区。
    *   对于CNN模型，可以点击“显示训练历史”来查看损失和准确率曲线。

4.  **识别人脸**:
    *   点击“选择图片”按钮，从您的电脑中选择一张包含人脸的图片。
    *   图片会显示在界面中央。
    *   点击“识别人脸”按钮，预测结果和置信度将显示在下方。

5.  **模型对比**:
    *   分别训练至少两个模型（例如，一个SVM模型和一个CNN模型）。
    *   点击“比较所有模型”按钮，一个清晰的性能对比表格将呈现在结果区。

## 📁 项目结构

```
FinalExp/
│
├── main.py                 # 程序主入口
│
├── gui/
│   └── main_window.py      # GUI界面实现
│
├── models/
│   ├── base_model.py       # 模型抽象基类
│   ├── svm_model.py        # SVM模型实现
│   └── cnn_model.py        # PyTorch CNN模型实现
│
├── utils/
│   ├── data_loader.py      # 数据加载与预处理
│   ├── face_detector.py    # 人脸检测器
│   └── evaluator.py        # 模型评估工具
│
├── Report/                   # 实验报告相关文件
│   └── Report.tex
│
└── requirements.txt          # 项目依赖包列表
