# main.py
"""
人脸识别系统主程序
支持SVM和CNN两种方法，提供图形化界面
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 确保能够导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    try:
        from gui.main_window import FaceRecognitionGUI
        
        # 创建并运行GUI应用
        app = FaceRecognitionGUI()
        app.run()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了所有依赖包")
        sys.exit(1)
    except Exception as e:
        print(f"程序运行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()