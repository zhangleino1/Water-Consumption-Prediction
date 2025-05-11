"""
训练并保存SVM模型用于水资源消耗预测API服务
"""

import os
import argparse
from ml_prediction_models import MLWaterConsumptionPredictor
from dataset import TimeSeriesDataModule

def train_svm_model(data_path="dataset", 
                     output_dir="ml_predictions", 
                     model_path=None, 
                     tune_hyperparameters=False,
                     province_id=None):
    """
    训练SVM模型并保存
    
    参数:
        data_path: 数据文件路径
        output_dir: 输出目录
        model_path: 模型保存路径
        tune_hyperparameters: 是否调优超参数
        province_id: 可选的特定省份ID
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果未指定模型路径，则使用默认路径
    if model_path is None:
        model_path = os.path.join(output_dir, "svm_model.joblib")
    
    print(f"将在 {model_path} 保存SVM模型")
    
    # 创建数据模块
    data_module = TimeSeriesDataModule(
        data_dir=data_path,
        sequence_length=5,
        batch_size=32,
        num_workers=0,
        province_id=province_id
    )
    
    # 创建SVM预测器
    predictor = MLWaterConsumptionPredictor(
        model_type='svm',
        sequence_length=5,
        output_dir=output_dir
    )
    
    print("准备数据...")
    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(data_module)
    
    print("训练SVM模型...")
    # 训练模型
    train_results = predictor.train(
        X_train, y_train, X_val, y_val,
        tune_hyperparameters=tune_hyperparameters
    )
    
    # 测试模型
    if len(X_test) > 0:
        print("在测试集上评估模型...")
        province_name = None
        if province_id is not None:
            # 尝试获取省份名称
            if data_module.full_dataset:
                for name, df in data_module.full_dataset.data.items():
                    if 'province_id' in df.columns and province_id in df['province_id'].values:
                        province_name = name
                        break
        
        test_results = predictor.test(
            X_test, y_test,
            province_name=province_name,
            province_id=province_id
        )
    
    # 保存模型
    print(f"保存模型到 {model_path}")
    predictor.save_model(model_path)
    print("模型训练和保存完成!")
    
    return {
        'train_results': train_results,
        'model_path': model_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练SVM模型用于水资源消耗预测API服务")
    parser.add_argument('--data_path', type=str, default='dataset',
                      help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='ml_predictions',
                      help='输出目录')
    parser.add_argument('--model_path', type=str, default=None,
                      help='模型保存路径 (默认为 output_dir/svm_model.joblib)')
    parser.add_argument('--tune_hyperparameters', action='store_true',
                      help='是否调优超参数')
    parser.add_argument('--province_id', type=int, default=None,
                      help='特定省份ID (默认为None，使用全国数据)')
    
    args = parser.parse_args()
    
    train_svm_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        tune_hyperparameters=args.tune_hyperparameters,
        province_id=args.province_id
    )
