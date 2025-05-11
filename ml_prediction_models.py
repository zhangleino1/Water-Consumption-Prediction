import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import argparse
from dataset import TimeSeriesDataModule  # 复用现有的数据加载模块

# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimSun'] 
matplotlib.rcParams['font.family'] = 'sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 


class MLWaterConsumptionPredictor:
    """
    使用传统机器学习模型（随机森林和支持向量机）进行水资源消耗预测
    """
    def __init__(self, model_type='rf', sequence_length=5, output_dir='ml_predictions'):
        """
        初始化预测器
        
        参数:
            model_type: 模型类型，'rf'代表随机森林，'svm'代表支持向量机
            sequence_length: 输入序列长度（用于特征生成）
            output_dir: 保存预测结果的目录
        """
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.output_dir = output_dir
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}。支持的类型有: 'rf', 'svm'")
    
    def prepare_data(self, data_module):
        """
        准备用于模型训练的数据
        
        参数:
            data_module: TimeSeriesDataModule实例
        
        返回:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # 确保数据模块已设置
        data_module.setup()
        
        # 获取训练集
        X_train, y_train = [], []
        train_loader = data_module.train_dataloader()
        for features, targets in train_loader:
            for feature, target in zip(features.numpy(), targets.numpy()):
                X_train.append(feature.flatten())  # 将序列展平为一维特征向量
                y_train.append(target.item())
        
        # 获取验证集
        X_val, y_val = [], []
        val_loader = data_module.val_dataloader()
        for features, targets in val_loader:
            for feature, target in zip(features.numpy(), targets.numpy()):
                X_val.append(feature.flatten())
                y_val.append(target.item())
        
        # 获取测试集
        X_test, y_test = [], []
        test_loader = data_module.test_dataloader()
        for features, targets in test_loader:
            for feature, target in zip(features.numpy(), targets.numpy()):
                X_test.append(feature.flatten())
                y_test.append(target.item())
        
        # 转换为NumPy数组
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        # 标准化特征 (对训练集拟合，对所有集合转换)
        X_train = self.scaler_X.fit_transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)
        
        # 标准化目标变量 (仅为训练计算，用于方便学习，但预测后会逆转换回原始值)
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train, X_val, y_val, tune_hyperparameters=False):
        """
        训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            tune_hyperparameters: 是否调优超参数
        """
        print(f"正在训练{self.model_type.upper()}模型...")
        
        if tune_hyperparameters:
            print("执行超参数调优...")
            
            if self.model_type == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                }
                
                grid_search = GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
            elif self.model_type == 'svm':
                param_grid = {
                    'kernel': ['linear', 'rbf', 'poly'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                    'epsilon': [0.01, 0.1, 0.2]
                }
                
                grid_search = GridSearchCV(
                    SVR(),
                    param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
            
            # 使用训练集和验证集合并进行网格搜索
            X_combined = np.concatenate([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            grid_search.fit(X_combined, y_combined)
            
            # 更新模型为最佳模型
            self.model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
        else:
            # 使用默认参数训练模型
            self.model.fit(X_train, y_train)
        
        # 在验证集上评估模型
        val_pred = self.model.predict(X_val)
        val_pred_original = self.scaler_y.inverse_transform(val_pred.reshape(-1, 1)).flatten()
        y_val_original = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        val_mse = mean_squared_error(y_val_original, val_pred_original)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val_original, val_pred_original)
        val_r2 = r2_score(y_val_original, val_pred_original)
        
        print(f"验证集性能:")
        print(f"MSE: {val_mse:.4f}")
        print(f"RMSE: {val_rmse:.4f}")
        print(f"MAE: {val_mae:.4f}")
        print(f"R²: {val_r2:.4f}")
        
        return {
            'mse': val_mse,
            'rmse': val_rmse,
            'mae': val_mae,
            'r2': val_r2
        }
    
    def test(self, X_test, y_test, province_name=None, province_id=None):
        """
        在测试集上评估模型
        
        参数:
            X_test: 测试特征
            y_test: 测试目标
            province_name: 可选的省份名称，用于保存结果
            province_id: 可选的省份ID
        """
        if self.model is None:
            raise ValueError("模型尚未训练。请先调用 train() 方法。")
        
        print(f"在测试集上评估{self.model_type.upper()}模型...")
        
        # 进行预测
        test_pred = self.model.predict(X_test)
        
        # 将预测值和真实值转换回原始尺度
        test_pred_original = self.scaler_y.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        y_test_original = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # 计算评估指标
        test_mse = mean_squared_error(y_test_original, test_pred_original)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test_original, test_pred_original)
        test_r2 = r2_score(y_test_original, test_pred_original)
        
        print(f"测试集性能:")
        print(f"MSE: {test_mse:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print(f"R²: {test_r2:.4f}")
        
        # 可视化预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_original, label='实际用水量', marker='o')
        plt.plot(test_pred_original, label='预测用水量', marker='x')
        plt.title(f'{self.model_type.upper()}模型：实际与预测水资源消耗对比')
        plt.xlabel('样本索引')
        plt.ylabel('水资源消耗量')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        province_str = f"_{province_name}" if province_name else ""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.output_dir, f'{self.model_type}_test_comparison{province_str}_{timestamp}.png'))
        plt.close()
        
        # 保存预测结果
        if province_name:
            results_df = pd.DataFrame({
                'Actual': y_test_original,
                'Predicted': test_pred_original,
                'Province': province_name,
                'Province_ID': province_id if province_id else 'Unknown'
            })
            
            csv_file = os.path.join(self.output_dir, f'{self.model_type}_{province_name}_predictions_{timestamp}.csv')
            results_df.to_csv(csv_file, index=False)
            print(f"预测结果已保存到 {csv_file}")
        
        return {
            'mse': test_mse,
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'predictions': test_pred_original,
            'actuals': y_test_original
        }
    
    def save_model(self, path=None):
        """
        保存训练好的模型
        
        参数:
            path: 可选的保存路径，如果未提供则使用默认路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练。请先调用 train() 方法。")
        
        # 如果没有提供路径，使用基于模型类型的默认路径
        if path is None:
            path = os.path.join(self.output_dir, f'{self.model_type}_model.joblib')
        
        # 保存模型和标准化器
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'model_type': self.model_type,
            'sequence_length': self.sequence_length
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump(model_data, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """
        加载预训练模型
        
        参数:
            path: 模型文件路径
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.model_type = model_data['model_type']
        self.sequence_length = model_data['sequence_length']
        
        print(f"已加载{self.model_type.upper()}模型 来自 {path}")
    
    def predict_future(self, data_module, years, province_id=None, province_name=None):
        """
        预测未来年份的水资源消耗
        
        参数:
            data_module: 数据模块
            years: 要预测的未来年份列表
            province_id: 可选的省份ID
            province_name: 可选的省份名称
        """
        if self.model is None:
            raise ValueError("模型尚未训练。请先调用 train() 方法。")
        
        # 确保数据模块已设置
        if not data_module.full_dataset:
            data_module.setup()
        
        # 如果年份是字符串，则转换为整数列表
        if isinstance(years, str):
            try:
                years = [int(y.strip()) for y in years.split(",")]
                print(f"预测年份: {years}")
            except ValueError:
                print("无效的年份格式。请提供逗号分隔的年份（例如，'2024,2025,2026'）")
                return
        
        # 获取对应省份的最新数据
        if province_id is None and province_name is None:
            print("必须提供省份ID或省份名称")
            return
        
        # 寻找对应省份的数据
        province_data = None
        actual_province_name = province_name if province_name else "未知"
        
        for name, df in data_module.full_dataset.data.items():
            # 按ID或名称匹配省份
            id_match = province_id is not None and 'province_id' in df.columns and province_id in df['province_id'].values
            name_match = province_name is not None and name.lower() == province_name.lower()
            
            if id_match or name_match:
                province_data = df
                actual_province_name = name
                break
        
        if province_data is None:
            print(f"未找到省份 ID:{province_id} 名称:{province_name} 的数据")
            return
        
        # 按年份排序
        province_data = province_data.sort_values('year')
        
        # 获取模型使用的特征列
        features = data_module.features
        
        # 获取最新的序列数据用于预测
        if len(province_data) < self.sequence_length:
            print(f"省份 {actual_province_name} 的历史数据不足")
            return
        
        # 提取最新的数据序列作为预测起点
        latest_data = province_data.iloc[-self.sequence_length:][features].values
        
        # 准备归一化年份特征
        future_years_norm = [(year - 2000) / 10 for year in years]
        
        # 保存历史数据用于作图
        historical_years = province_data['year'].tolist()
        historical_consumption = province_data['consumption'].tolist()
        
        # 对于每个未来年份，生成预测
        predictions = []
        
        # 复制最新的序列作为起点
        prediction_sequence = latest_data.copy()
        
        for i, year in enumerate(years):
            # 展平序列作为模型输入
            X_pred = prediction_sequence.flatten().reshape(1, -1)
            
            # 标准化输入特征
            X_pred = self.scaler_X.transform(X_pred)
            
            # 进行预测
            pred = self.model.predict(X_pred)
            
            # 将预测结果转换回原始尺度
            pred_original = self.scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()[0]
            predictions.append(pred_original)
            
            # 更新预测序列（移除最早的时间步，添加新的预测）
            prediction_sequence = np.roll(prediction_sequence, -1, axis=0)
            
            # 更新最后一行的特征（使用最新一行的特征作为模板）
            prediction_sequence[-1] = prediction_sequence[-2].copy()
            
            # 更新年份特征
            year_idx = 2  # 假设归一化年份是特征列表中的第3个（索引2）
            if prediction_sequence.shape[1] > year_idx:
                prediction_sequence[-1, year_idx] = (year - 2000) / 10
        
        # 创建结果数据框
        results = pd.DataFrame({
            'Year': years,
            'Predicted_Consumption': predictions
        })
        
        print(f"\n{actual_province_name} (ID: {province_id}) 的未来水资源消耗预测:")
        print(results.to_string(index=False))
        
        # 可视化结果
        try:
            plt.figure(figsize=(12, 6))
            
            # 绘制历史数据
            plt.plot(historical_years, historical_consumption, 'o-', label='历史数据')
            
            # 绘制预测
            plt.plot(years, predictions, 'x--', color='red', label=f'{self.model_type.upper()}预测')
            
            # 添加标签和标题
            plt.xlabel('年份')
            plt.ylabel('水资源消耗量')
            plt.title(f'{actual_province_name} (ID: {province_id}) 的水资源消耗预测')
            plt.grid(True)
            plt.legend()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(self.output_dir, f'{self.model_type}_{actual_province_name}_id{province_id}_{timestamp}.png'))
            print(f"预测图表已保存到 {self.output_dir}")
            
            plt.close()
        except Exception as e:
            print(f"可视化预测时出错: {e}")
        
        # 将预测结果保存到CSV
        try:
            # 创建包含历史和预测数据的数据框
            all_data = pd.DataFrame({
                'Year': historical_years + years,
                'Consumption': historical_consumption + [None] * len(years),
                'Predicted': [None] * len(historical_years) + predictions,
                'Province': actual_province_name,
                'Province_ID': province_id,
                'Model': self.model_type.upper()
            })
            
            # 保存到CSV
            csv_file = os.path.join(self.output_dir, f'{self.model_type}_{actual_province_name}_id{province_id}_{timestamp}.csv')
            all_data.to_csv(csv_file, index=False)
            print(f"预测数据已保存到 {csv_file}")
        except Exception as e:
            print(f"保存预测数据时出错: {e}")
        
        return predictions


def train_model(args):
    """训练模型的主函数"""
    # 创建数据模块
    data_module = TimeSeriesDataModule(
        data_dir=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        province_id=args.province_id
    )
    
    # 创建ML预测器
    predictor = MLWaterConsumptionPredictor(
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir
    )
    
    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(data_module)
    
    # 训练模型
    train_results = predictor.train(
        X_train, y_train, X_val, y_val,
        tune_hyperparameters=args.tune_hyperparameters
    )
    
    # 测试模型
    province_name = None
    if len(X_test) > 0:
        if args.province_id is not None:
            # 尝试获取省份名称
            if data_module.full_dataset:
                for name, df in data_module.full_dataset.data.items():
                    if 'province_id' in df.columns and args.province_id in df['province_id'].values:
                        province_name = name
                        break
        
        test_results = predictor.test(
            X_test, y_test,
            province_name=province_name or args.province_name,
            province_id=args.province_id
        )
    
    # 根据省份ID生成模型保存路径，确保不同省份的模型不会互相覆盖
    if args.province_id is not None:
        # 构建包含省份ID的模型文件名
        model_filename = f"{args.model_type}_province_{args.province_id}_model.joblib"
        model_path = os.path.join(args.output_dir, model_filename)
    else:
        # 如果没有指定省份ID，则使用通用模型名称（用于全国模型）
        model_path = args.model_path
    
    # 保存模型
    predictor.save_model(model_path)
    print(f"已保存模型到: {model_path}")
    
    # 如果指定了预测年份，则进行未来预测
    if args.predict_years and (args.province_id is not None or args.province_name is not None):
        predictor.predict_future(
            data_module=data_module,
            years=args.predict_years,
            province_id=args.province_id,
            province_name=args.province_name or province_name
        )


def test_model(args):
    """测试预训练模型的主函数"""
    # 创建ML预测器并加载模型
    predictor = MLWaterConsumptionPredictor(
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir
    )
    
    predictor.load_model(args.model_path)
    
    # 创建数据模块
    data_module = TimeSeriesDataModule(
        data_dir=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        province_id=args.province_id
    )
    
    # 准备数据
    _, _, X_test, _, _, y_test = predictor.prepare_data(data_module)
    
    # 如果有测试数据，则进行测试
    if len(X_test) > 0:
        province_name = None
        if args.province_id is not None:
            # 尝试获取省份名称
            if data_module.full_dataset:
                for name, df in data_module.full_dataset.data.items():
                    if 'province_id' in df.columns and args.province_id in df['province_id'].values:
                        province_name = name
                        break
        
        predictor.test(
            X_test, y_test,
            province_name=province_name or args.province_name,
            province_id=args.province_id
        )
    
    # 如果指定了预测年份，则进行未来预测
    if args.predict_years and (args.province_id is not None or args.province_name is not None):
        predictor.predict_future(
            data_module=data_module,
            years=args.predict_years,
            province_id=args.province_id,
            province_name=args.province_name
        )


def list_provinces(args):
    """列出数据集中所有可用的省份及其ID"""
    # 创建临时数据模块以访问省份信息
    data_module = TimeSeriesDataModule(data_dir=args.data_path)
    
    # 强制设置以加载数据
    data_module.setup()
    
    # 获取并打印省份信息
    if data_module.full_dataset and hasattr(data_module.full_dataset, 'data'):
        # 加载合并的数据文件以获取省份ID和名称
        try:
            csv_path = os.path.join(args.data_path, 'merged_water_data.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                province_info = df[['province_id', 'province']].drop_duplicates()
                print("\n可用省份:")
                print("-------------------")
                for _, row in province_info.sort_values('province_id').iterrows():
                    print(f"ID: {row['province_id']}, 名称: {row['province']}")
            else:
                print("找不到merged_water_data.csv文件。")
        except Exception as e:
            print(f"加载省份信息时出错: {e}")
            # 回退到仅显示可用省份而不显示ID
            print("\n可用省份 (仅名称):")
            for province in data_module.get_province_names():
                print(f"- {province}")
    else:
        print("没有可用的省份数据。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用随机森林和支持向量机模型进行水资源消耗预测")
    
    # 基本参数
    parser.add_argument('--model_type', type=str, choices=['rf', 'svm'], default='rf',
                      help='模型类型: rf (随机森林) 或 svm (支持向量机)')
    parser.add_argument('--data_path', type=str, default='dataset',
                      help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='ml_predictions',
                      help='保存预测结果的目录')
    parser.add_argument('--model_path', type=str, default=None,
                      help='模型保存/加载路径 (默认为 output_dir/model_type_model.joblib)')
    
    # 数据和训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='数据加载器使用的工作线程数')
    parser.add_argument('--sequence_length', type=int, default=5,
                      help='输入序列长度')
    parser.add_argument('--tune_hyperparameters', action='store_true',
                      help='是否执行超参数调优')
    
    # 省份和预测参数
    parser.add_argument('--province_id', type=int, default=410000,
                      help='要训练/预测的省份ID')
    parser.add_argument('--province_name', type=str, default='河南',
                      help='要训练/预测的省份名称 (如果未指定province_id)')
    parser.add_argument('--predict_years', type=str, default="2024,2025,2026",
                      help='预测水资源消耗的年份，以逗号分隔')
    
    # 操作模式
    parser.add_argument('--mode', choices=['train', 'test', 'list_provinces'], 
                      type=str, default='train',
                      help='操作模式: train (训练), test (测试), list_provinces (列出省份)')
    
    args = parser.parse_args()
    
    # 如果未指定模型路径，则使用默认路径
    if args.model_path is None:
        args.model_path = os.path.join(args.output_dir, f'{args.model_type}_model.joblib')
    
    # 处理省份ID（如果以字符串形式提供）
    if hasattr(args, 'province_id') and args.province_id is not None:
        if isinstance(args.province_id, str):
            try:
                args.province_id = int(args.province_id)
            except ValueError:
                print(f"警告：无效的province_id格式：{args.province_id}。使用None代替。")
                args.province_id = None
    
    # 以指定模式运行
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'list_provinces':
        list_provinces(args)
    else:
        print(f"未知模式：{args.mode}")