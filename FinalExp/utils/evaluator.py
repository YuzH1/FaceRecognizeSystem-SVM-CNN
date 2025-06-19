# utils/evaluator.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import pandas as pd

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        pass
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None, 
                      class_names: List[str] = None) -> Dict[str, Any]:
        """全面评估模型性能"""
        
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 详细分类报告
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names if class_names else list(set(y_true))
        }
        
        return results
    
    def print_evaluation(self, results: Dict[str, Any]):
        """打印评估结果"""
        print("=" * 50)
        print("模型评估结果")
        print("=" * 50)
        print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
        print(f"精确率 (Precision - Macro): {results['precision_macro']:.4f}")
        print(f"召回率 (Recall - Macro): {results['recall_macro']:.4f}")
        print(f"F1分数 (F1-Score - Macro): {results['f1_macro']:.4f}")
        print(f"精确率 (Precision - Micro): {results['precision_micro']:.4f}")
        print(f"召回率 (Recall - Micro): {results['recall_micro']:.4f}")
        print(f"F1分数 (F1-Score - Micro): {results['f1_micro']:.4f}")
        print("\n详细分类报告:")
        
        # 格式化分类报告
        df_report = pd.DataFrame(results['classification_report']).transpose()
        print(df_report.round(4))
    
    def plot_confusion_matrix(self, results: Dict[str, Any], title: str = "混淆矩阵"):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=results['class_names'],
                   yticklabels=results['class_names'])
        plt.title(title)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """比较多个模型的性能"""
        comparison_data = []
        
        for model_name, results in results_dict.items():
            comparison_data.append({
                '模型': model_name,
                '准确率': results['accuracy'],
                '精确率(宏平均)': results['precision_macro'],
                '召回率(宏平均)': results['recall_macro'],
                'F1分数(宏平均)': results['f1_macro'],
                '精确率(微平均)': results['precision_micro'],
                '召回率(微平均)': results['recall_micro'],
                'F1分数(微平均)': results['f1_micro']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison.round(4)