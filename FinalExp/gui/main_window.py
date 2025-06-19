# gui/main_window.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
from typing import Dict, Any

class FaceRecognitionGUI:
    """人脸识别系统图形界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("人脸识别系统")
        self.root.geometry("1200x800")
        
        # 模型相关
        self.models = {}
        self.current_model = None
        self.data_loader = None
        self.face_detector = None
        self.evaluator = None
        
        # 数据相关
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_names = None
        
        # 图像显示
        self.current_image = None
        self.display_image = None
        
        self.setup_gui()
        self.setup_models()
    
    def setup_models(self):
        """初始化模型"""
        from models.svm_model import SVMFaceRecognitionModel
        from models.cnn_model import PyTorchCNNFaceRecognitionModel  # 新增
        from utils.data_loader import FaceDataLoader
        from utils.face_detector import FaceDetector
        from utils.evaluator import ModelEvaluator
        
        self.models = {
            'SVM': SVMFaceRecognitionModel(),
            'PyTorch CNN': PyTorchCNNFaceRecognitionModel()  # 替换原来的CNN
        }
        
        self.face_detector = FaceDetector()
        self.evaluator = ModelEvaluator()
    
    def setup_gui(self):
        """设置GUI布局"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 左侧控制面板
        self.setup_control_panel(main_frame)
        
        # 右侧显示区域
        self.setup_display_area(main_frame)
        
        # 底部状态栏
        self.setup_status_bar(main_frame)
    
    def setup_control_panel(self, parent):
        """设置控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 数据加载部分
        data_frame = ttk.LabelFrame(control_frame, text="数据管理", padding="5")
        data_frame.pack(fill="x", pady=(0, 10))
        
        # 数据集选择
        dataset_select_frame = ttk.Frame(data_frame)
        dataset_select_frame.pack(fill="x", pady=2)
        
        ttk.Label(dataset_select_frame, text="选择数据集:").pack(side="left")
        self.dataset_var = tk.StringVar(value="olivetti")
        dataset_combo = ttk.Combobox(dataset_select_frame, textvariable=self.dataset_var,
                                    values=["olivetti", "lfw", "custom"], state="readonly")
        dataset_combo.pack(side="right", padx=(5, 0))
        
        # 加载按钮
        button_frame = ttk.Frame(data_frame)
        button_frame.pack(fill="x", pady=2)
        
        ttk.Button(button_frame, text="加载内置数据集", 
                command=self.load_builtin_dataset).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="选择自定义文件夹", 
                command=self.load_custom_dataset).pack(side="left")
        
        # 数据集信息显示
        self.data_info_label = ttk.Label(data_frame, text="未加载数据集")
        self.data_info_label.pack(fill="x", pady=2)
        
        ttk.Button(data_frame, text="查看数据样本", 
                command=self.show_data_samples).pack(fill="x", pady=2)
        
        # 模型选择部分
        model_frame = ttk.LabelFrame(control_frame, text="模型选择", padding="5")
        model_frame.pack(fill="x", pady=(0, 10))
        
        self.model_var = tk.StringVar(value="SVM")
        ttk.Radiobutton(model_frame, text="SVM模型", 
                    variable=self.model_var, value="SVM").pack(anchor="w")
        ttk.Radiobutton(model_frame, text="PyTorch CNN模型", 
                    variable=self.model_var, value="PyTorch CNN").pack(anchor="w")
        # 添加模型参数设置
        params_frame = ttk.LabelFrame(model_frame, text="模型参数", padding="3")
        params_frame.pack(fill="x", pady=(5, 0))
        
        # 学习率设置
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill="x", pady=2)
        ttk.Label(lr_frame, text="学习率:").pack(side="left")
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=10).pack(side="right")
        
        # 批次大小设置
        batch_frame = ttk.Frame(params_frame)
        batch_frame.pack(fill="x", pady=2)
        ttk.Label(batch_frame, text="批次大小:").pack(side="left")
        self.batch_var = tk.StringVar(value="32")
        ttk.Entry(batch_frame, textvariable=self.batch_var, width=10).pack(side="right")
        
        # 训练轮数设置
        epochs_frame = ttk.Frame(params_frame)
        epochs_frame.pack(fill="x", pady=2)
        ttk.Label(epochs_frame, text="训练轮数:").pack(side="left")
        self.epochs_var = tk.StringVar(value="30")
        ttk.Entry(epochs_frame, textvariable=self.epochs_var, width=10).pack(side="right")

        # 模型操作部分
        operation_frame = ttk.LabelFrame(control_frame, text="模型操作", padding="5")
        operation_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(operation_frame, text="训练模型", 
                command=self.train_model).pack(fill="x", pady=2)
        ttk.Button(operation_frame, text="评估模型", 
                command=self.evaluate_model).pack(fill="x", pady=2)
        ttk.Button(operation_frame, text="显示训练历史", 
                command=self.show_training_history).pack(fill="x", pady=2)  # 新增
        ttk.Button(operation_frame, text="保存模型", 
                command=self.save_model).pack(fill="x", pady=2)
        ttk.Button(operation_frame, text="加载模型", 
                command=self.load_model).pack(fill="x", pady=2)
        
        # 预测部分
        predict_frame = ttk.LabelFrame(control_frame, text="人脸识别", padding="5")
        predict_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(predict_frame, text="选择图片", 
                  command=self.select_image).pack(fill="x", pady=2)
        ttk.Button(predict_frame, text="识别人脸", 
                  command=self.predict_face).pack(fill="x", pady=2)
        
        self.prediction_label = ttk.Label(predict_frame, text="预测结果将在此显示")
        self.prediction_label.pack(fill="x", pady=2)
        
        # 比较模型部分
        compare_frame = ttk.LabelFrame(control_frame, text="模型比较", padding="5")
        compare_frame.pack(fill="x")
        
        ttk.Button(compare_frame, text="比较所有模型", 
                  command=self.compare_models).pack(fill="x", pady=2)
    

    def load_custom_dataset(self):
        """加载自定义数据集"""
        folder_path = filedialog.askdirectory(title="选择数据集文件夹")
        if not folder_path:
            return
        
        try:
            self.update_status("正在加载自定义数据集...")
            
            from utils.data_loader import FaceDataLoader
            self.data_loader = FaceDataLoader("custom", data_dir=folder_path)
            
            # 在新线程中加载数据
            def load_data():
                try:
                    self.X_train, self.X_test, self.y_train, self.y_test, self.class_names = \
                        self.data_loader.load_dataset()
                    
                    # 更新UI
                    self.root.after(0, self.on_data_loaded)
                except Exception as e:
                    self.root.after(0, lambda: self.on_error(f"加载自定义数据集失败: {str(e)}"))
            
            threading.Thread(target=load_data, daemon=True).start()
            
        except Exception as e:
            self.on_error(f"加载自定义数据集失败: {str(e)}")

   
    def load_builtin_dataset(self):
        """加载内置数据集"""
        dataset_type = self.dataset_var.get()
        
        try:
            self.update_status(f"正在加载{dataset_type}数据集...")
            
            # 在新线程中加载数据
            def load_data():
                try:
                    from utils.data_loader import FaceDataLoader
                    
                    if dataset_type == "olivetti":
                        self.data_loader = FaceDataLoader("olivetti")
                    elif dataset_type == "lfw":
                        self.data_loader = FaceDataLoader("lfw")
                        
                    self.X_train, self.X_test, self.y_train, self.y_test, self.class_names = \
                        self.data_loader.load_dataset()
                    
                    # 更新UI
                    self.root.after(0, self.on_data_loaded)
                except Exception as e:
                    self.root.after(0, lambda: self.on_error(f"加载{dataset_type}数据集失败: {str(e)}"))
            
            threading.Thread(target=load_data, daemon=True).start()
            
        except Exception as e:
            self.on_error(f"加载{dataset_type}数据集失败: {str(e)}")   



    def show_data_samples(self):
        """显示数据样本"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据集")
            return
        
        try:
            # 在新线程中显示样本
            def show_samples():
                self.data_loader.visualize_samples(
                    self.X_train, self.y_train, self.class_names, n_samples=12
                )
            
            threading.Thread(target=show_samples, daemon=True).start()
            
        except Exception as e:
            self.on_error(f"显示数据样本失败: {str(e)}")

    def setup_display_area(self, parent):
        """设置显示区域"""
        display_frame = ttk.LabelFrame(parent, text="显示区域", padding="10")
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 图像显示区域
        self.image_label = ttk.Label(display_frame, text="图像将在此显示")
        self.image_label.pack(pady=10)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(display_frame, text="详细结果", padding="5")
        result_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        # 创建滚动文本框
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, height=20)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_status_bar(self, parent):
        """设置状态栏"""
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.root.update()
    
    def load_dataset(self):
        """加载数据集"""
        folder_path = filedialog.askdirectory(title="选择数据集文件夹")
        if not folder_path:
            return
        
        try:
            self.update_status("正在加载数据集...")
            
            from utils.data_loader import FaceDataLoader
            self.data_loader = FaceDataLoader(folder_path)
            
            # 在新线程中加载数据
            def load_data():
                try:
                    self.X_train, self.X_test, self.y_train, self.y_test, self.class_names = \
                        self.data_loader.load_dataset()
                    
                    # 更新UI
                    self.root.after(0, self.on_data_loaded)
                except Exception as e:
                    self.root.after(0, lambda: self.on_error(f"加载数据集失败: {str(e)}"))
            
            threading.Thread(target=load_data, daemon=True).start()
            
        except Exception as e:
            self.on_error(f"加载数据集失败: {str(e)}")
    
    def on_data_loaded(self):
        """数据加载完成后的回调"""
        info_text = f"数据集已加载\n训练集: {len(self.X_train)} 样本\n测试集: {len(self.X_test)} 样本\n类别数: {len(self.class_names)}"
        self.data_info_label.config(text=info_text)
        self.update_status("数据集加载完成")
        
        # 显示数据集信息
        self.append_result(f"数据集信息:\n{info_text}\n类别: {', '.join(self.class_names)}\n")
    
    def train_model(self):
        """训练选定的模型"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据集")
            return
        
        model_name = self.model_var.get()
        model = self.models[model_name]
        
        try:
            self.update_status(f"正在训练{model_name}模型...")
            
            # 获取训练参数
            learning_rate = float(self.lr_var.get()) if self.lr_var.get() else 0.001
            batch_size = int(self.batch_var.get()) if self.batch_var.get() else 32
            epochs = int(self.epochs_var.get()) if self.epochs_var.get() else 30
            
            # 在新线程中训练模型
            def train():
                try:
                    if model_name == "PyTorch CNN":
                        # 更新PyTorch模型参数
                        model.learning_rate = learning_rate
                        model.batch_size = batch_size
                        
                        # PyTorch CNN需要验证集
                        train_info = model.train(
                            self.X_train, self.y_train, 
                            self.X_test, self.y_test, 
                            epochs=epochs, 
                            batch_size=batch_size
                        )
                    else:
                        # SVM模型
                        train_info = model.train(self.X_train, self.y_train)
                    
                    # 更新UI
                    self.root.after(0, lambda: self.on_model_trained(model_name, train_info))
                except Exception as e:
                    self.root.after(0, lambda: self.on_error(f"训练{model_name}模型失败: {str(e)}"))
            
            threading.Thread(target=train, daemon=True).start()
            
        except ValueError as e:
            self.on_error(f"参数错误: {str(e)}")
        except Exception as e:
            self.on_error(f"训练{model_name}模型失败: {str(e)}")
    
    def on_model_trained(self, model_name, train_info):
        """模型训练完成后的回调"""
        self.current_model = self.models[model_name]
        self.update_status(f"{model_name}模型训练完成")
        
        # 显示训练信息
        info_text = f"{model_name}模型训练完成:\n"
        for key, value in train_info.items():
            info_text += f"{key}: {value}\n"
        
        self.append_result(info_text + "\n")

    def show_training_history(self):
        """显示训练历史（仅适用于PyTorch CNN）"""
        if self.current_model is None:
            messagebox.showerror("错误", "请先训练模型")
            return
        
        model_name = self.model_var.get()
        if model_name == "PyTorch CNN" and hasattr(self.current_model, 'plot_training_history'):
            try:
                def plot_history():
                    self.current_model.plot_training_history()
                
                threading.Thread(target=plot_history, daemon=True).start()
            except Exception as e:
                self.on_error(f"显示训练历史失败: {str(e)}")
        else:
            messagebox.showinfo("信息", "此功能仅适用于PyTorch CNN模型")
    
    def evaluate_model(self):
        """评估当前模型"""
        if self.current_model is None:
            messagebox.showerror("错误", "请先训练模型")
            return
        
        if self.X_test is None:
            messagebox.showerror("错误", "请先加载数据集")
            return
        
        try:
            self.update_status("正在评估模型...")
            
            # 在新线程中评估模型
            def evaluate():
                try:
                    # 预测
                    y_pred = self.current_model.predict(self.X_test)
                    y_pred_proba = self.current_model.predict_proba(self.X_test)
                    
                    # 评估
                    results = self.evaluator.evaluate_model(
                        self.y_test, y_pred, y_pred_proba, self.class_names)
                    
                    # 更新UI
                    self.root.after(0, lambda: self.on_model_evaluated(results))
                except Exception as e:
                    self.root.after(0, lambda: self.on_error(f"评估模型失败: {str(e)}"))
            
            threading.Thread(target=evaluate, daemon=True).start()
            
        except Exception as e:
            self.on_error(f"评估模型失败: {str(e)}")
    
    def on_model_evaluated(self, results):
        """模型评估完成后的回调"""
        self.update_status("模型评估完成")
        
        # 格式化评估结果
        eval_text = "模型评估结果:\n"
        eval_text += "=" * 50 + "\n"
        eval_text += f"准确率: {results['accuracy']:.4f}\n"
        eval_text += f"精确率(宏平均): {results['precision_macro']:.4f}\n"
        eval_text += f"召回率(宏平均): {results['recall_macro']:.4f}\n"
        eval_text += f"F1分数(宏平均): {results['f1_macro']:.4f}\n"
        eval_text += f"精确率(微平均): {results['precision_micro']:.4f}\n"
        eval_text += f"召回率(微平均): {results['recall_micro']:.4f}\n"
        eval_text += f"F1分数(微平均): {results['f1_micro']:.4f}\n"
        eval_text += "=" * 50 + "\n\n"
        
        self.append_result(eval_text)
    
    def save_model(self):
        """保存当前模型"""
        if self.current_model is None:
            messagebox.showerror("错误", "没有可保存的模型")
            return
        
        model_name = self.model_var.get()
        file_path = filedialog.asksaveasfilename(
            title="保存模型",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_model.save_model(file_path)
                self.update_status(f"{model_name}模型已保存")
                messagebox.showinfo("成功", f"{model_name}模型已保存到 {file_path}")
            except Exception as e:
                self.on_error(f"保存模型失败: {str(e)}")
    
    def load_model(self):
        """加载模型"""
        file_path = filedialog.askopenfilename(
            title="加载模型",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                model_name = self.model_var.get()
                self.models[model_name].load_model(file_path)
                self.current_model = self.models[model_name]
                self.update_status(f"{model_name}模型已加载")
                messagebox.showinfo("成功", f"{model_name}模型已从 {file_path} 加载")
            except Exception as e:
                self.on_error(f"加载模型失败: {str(e)}")
    
    def select_image(self):
        """选择要识别的图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 读取并显示图片
                self.current_image = cv2.imread(file_path)
                self.display_selected_image()
                self.update_status("图片已选择")
            except Exception as e:
                self.on_error(f"加载图片失败: {str(e)}")
    
    def display_selected_image(self):
        """显示选择的图片"""
        if self.current_image is not None:
            # 调整图片大小用于显示
            display_img = self.current_image.copy()
            height, width = display_img.shape[:2]
            
            # 限制显示大小
            max_size = 400
            if height > max_size or width > max_size:
                scale = min(max_size/height, max_size/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_img = cv2.resize(display_img, (new_width, new_height))
            
            # 转换为RGB并显示
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_img_rgb)
            self.display_image = ImageTk.PhotoImage(pil_image)
            
            self.image_label.config(image=self.display_image, text="")
    
    def predict_face(self):
        """识别选定图片中的人脸"""
        if self.current_image is None:
            messagebox.showerror("错误", "请先选择图片")
            return
        
        if self.current_model is None:
            messagebox.showerror("错误", "请先训练或加载模型")
            return
        
        try:
            self.update_status("正在识别人脸...")
            
            # 提取人脸
            face = self.face_detector.extract_face(self.current_image, size=(64, 64))
            
            if face is None:
                messagebox.showwarning("警告", "未检测到人脸")
                self.update_status("未检测到人脸")
                return
            
            # 预处理
            if len(face.shape) == 2:
                face_input = np.expand_dims(face, axis=(0, -1))  # 添加batch和通道维度
            else:
                face_input = np.expand_dims(face, axis=0)  # 添加batch维度
            
            # 预测
            prediction = self.current_model.predict(face_input)[0]
            probabilities = self.current_model.predict_proba(face_input)[0]
            
            # 显示结果
            confidence = np.max(probabilities)
            
            result_text = f"预测结果: {prediction}\n置信度: {confidence:.4f}\n\n"
            result_text += "所有类别的概率:\n"
            
            # 获取类别名称
            if hasattr(self.current_model, 'label_encoder'):
                class_names = self.current_model.label_encoder.classes_
            else:
                class_names = self.class_names
            
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                result_text += f"{class_name}: {prob:.4f}\n"
            
            self.prediction_label.config(text=f"预测: {prediction} (置信度: {confidence:.4f})")
            self.append_result(result_text + "\n")
            self.update_status("人脸识别完成")
            
        except Exception as e:
            self.on_error(f"人脸识别失败: {str(e)}")
    
    def compare_models(self):
        """比较所有已训练的模型"""
        if self.X_test is None:
            messagebox.showerror("错误", "请先加载数据集")
            return
        
        # 检查哪些模型已训练
        trained_models = {name: model for name, model in self.models.items() if model.is_trained}
        
        if len(trained_models) < 2:
            messagebox.showwarning("警告", "请至少训练两个模型进行比较")
            return
        
        try:
            self.update_status("正在比较模型...")
            
            def compare():
                try:
                    results_dict = {}
                    
                    for model_name, model in trained_models.items():
                        # 预测
                        y_pred = model.predict(self.X_test)
                        y_pred_proba = model.predict_proba(self.X_test)
                        
                        # 评估
                        results = self.evaluator.evaluate_model(
                            self.y_test, y_pred, y_pred_proba, self.class_names)
                        results_dict[model_name] = results
                    
                    # 更新UI
                    self.root.after(0, lambda: self.on_models_compared(results_dict))
                except Exception as e:
                    self.root.after(0, lambda: self.on_error(f"模型比较失败: {str(e)}"))
            
            threading.Thread(target=compare, daemon=True).start()
            
        except Exception as e:
            self.on_error(f"模型比较失败: {str(e)}")
    
    def on_models_compared(self, results_dict):
        """模型比较完成后的回调"""
        self.update_status("模型比较完成")
        
        # 创建比较表格
        comparison_df = self.evaluator.compare_models(results_dict)
        
        comparison_text = "模型性能比较:\n"
        comparison_text += "=" * 80 + "\n"
        comparison_text += comparison_df.to_string(index=False) + "\n"
        comparison_text += "=" * 80 + "\n\n"
        
        self.append_result(comparison_text)
    
    def append_result(self, text):
        """在结果显示区域添加文本"""
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)
    
    def on_error(self, error_message):
        """错误处理"""
        self.update_status("发生错误")
        messagebox.showerror("错误", error_message)
        self.append_result(f"错误: {error_message}\n\n")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()

# 创建并运行GUI
if __name__ == "__main__":
    app = FaceRecognitionGUI()
    app.run()