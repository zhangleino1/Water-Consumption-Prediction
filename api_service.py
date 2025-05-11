import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import joblib
import uvicorn
from ml_prediction_models import MLWaterConsumptionPredictor
from dataset import TimeSeriesDataModule

# 创建FastAPI实例
app = FastAPI(
    title="水资源消耗预测API",
    description="提供省级和全国水资源消耗历史数据及基于SVM模型的预测",
    version="1.0.0"
)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 定义数据模型
class PredictionRequest(BaseModel):
    years: List[int] = Field(default=[2024, 2025, 2026], description="需要预测的年份列表")
    province_id: Optional[int] = Field(default=None, description="省份ID，如果不提供则预测全国数据")
    province_name: Optional[str] = Field(default=None, description="省份名称（如果同时提供省份ID和名称，优先使用ID）")
    model_type: str = Field(default="svm", description="预测模型类型：'svm'(支持向量机)、'rf'(随机森林)或'simulation'(模拟预测)")

class DataPoint(BaseModel):
    year: int = Field(..., description="年份")
    consumption: Optional[float] = Field(None, description="实际水资源消耗量")
    predicted: Optional[float] = Field(None, description="预测的水资源消耗量")

class PredictionResponse(BaseModel):
    province_id: Optional[int] = Field(None, description="省份ID")
    province_name: str = Field(..., description="省份名称")
    data: List[DataPoint] = Field(..., description="历史和预测数据")
    last_trained: str = Field(..., description="模型最后训练日期")
    model_type: str = Field(..., description="使用的模型类型")

# 全局变量
# 基础目录
ML_PREDICTIONS_DIR = "ml_predictions"
DATA_PATH = "dataset"
SEQUENCE_LENGTH = 5

# 初始化数据模块和预测器
data_module = None
predictors = {}

def get_model_path(model_type, province_id=None):
    """
    根据模型类型和省份ID获取模型文件路径
    
    参数:
        model_type: 模型类型（'svm', 'rf'）
        province_id: 省份ID，如果为None则使用全国模型
    
    返回:
        str: 模型文件路径
    """
    # 验证模型类型
    if model_type not in ['svm', 'rf']:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 构建模型文件名
    if province_id:
        model_file = f"{model_type}_model_province_{province_id}.joblib"
    else:
        model_file = f"{model_type}_model.joblib"
    
    return os.path.join(ML_PREDICTIONS_DIR, model_file)

def load_models():
    """加载数据和模型"""
    global  predictors
    
   
    
    # 为svm和rf模型类型创建预测器
    for model_type in ['svm', 'rf']:
        # 尝试加载全国模型（不指定province_id）
        try:
            model_path = get_model_path(model_type)
            predictor = MLWaterConsumptionPredictor(
                model_type=model_type,
                sequence_length=SEQUENCE_LENGTH
            )
            predictor.load_model(model_path)
            key = f"{model_type}_0"  # 0表示全国
            predictors[key] = predictor
            print(f"成功加载{model_type.upper()}全国模型: {model_path}")
        except Exception as e:
            print(f"无法加载{model_type.upper()}全国模型: {e}")

    # 扫描ml_predictions目录，查找具有province_id的模型
    try:
        for filename in os.listdir(ML_PREDICTIONS_DIR):
            if filename.endswith(".joblib"):
                # 检查是否为具有省份ID的模型
                if "_model_province_" in filename:
                    parts = filename.split("_model_province_")
                    if len(parts) == 2:
                        model_type = parts[0]
                        province_id = int(parts[1].replace(".joblib", ""))
                        
                        # 检查是否为支持的模型类型
                        if model_type in ['svm', 'rf']:
                            try:
                                model_path = os.path.join(ML_PREDICTIONS_DIR, filename)
                                predictor = MLWaterConsumptionPredictor(
                                    model_type=model_type,
                                    sequence_length=SEQUENCE_LENGTH
                                )
                                predictor.load_model(model_path)
                                key = f"{model_type}_{province_id}"
                                predictors[key] = predictor
                                print(f"成功加载{model_type.upper()}省份模型 (ID: {province_id}): {model_path}")
                            except Exception as e:
                                print(f"无法加载{model_type.upper()}省份模型 (ID: {province_id}): {e}")
    except Exception as e:
        print(f"扫描模型目录时出错: {e}")

    # 如果没有任何模型被加载，创建默认SVM模型
    if not predictors:
        print("未加载任何模型，将创建默认SVM模型")
        predictor = MLWaterConsumptionPredictor(
            model_type='svm',
            sequence_length=SEQUENCE_LENGTH
        )
        predictors['svm_0'] = predictor

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    try:
        load_models()
    except Exception as e:
        print(f"启动时加载模型出错: {e}")

def get_model_info(model_type='svm', province_id=None):
    """
    获取指定模型的信息
    
    参数:
        model_type: 模型类型（'svm', 'rf', 'simulation'）
        province_id: 省份ID
    
    返回:
        dict: 包含模型类型和最后训练时间的字典
    """
    # 对于模拟预测，直接返回固定信息
    if model_type == 'simulation':
        return {
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "模拟预测"
        }
    
    try:
        # 获取实际模型文件路径
        if model_type in ['svm', 'rf']:
            model_path = get_model_path(model_type, province_id)
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                last_modified = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
                model_type_display = model_data.get('model_type', model_type).upper()
                return {
                    "last_trained": last_modified,
                    "model_type": model_type_display
                }
            
        # 如果模型文件不存在，查找匹配的省份模型
        key = f"{model_type}_{province_id if province_id is not None else 0}"
        if key in predictors:
            return {
                "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_type": model_type.upper()
            }
    
    except Exception as e:
        print(f"获取模型信息出错: {e}")
    
    # 默认返回
    return {
        "last_trained": "未训练", 
        "model_type": f"{model_type.upper()} (未训练)"
    }

def simulate_predictions(historical_data, future_years):
    """
    使用模拟算法预测未来用水量数据，基于plot_water_consumption.py中的方法
    
    参数:
        historical_data (DataFrame): 历史用水量数据，包含'year'和'consumption'列
        future_years (list): 要预测的未来年份列表
    
    返回:
        list: 预测的用水量数据
    """
    # 提取最近5年的数据以计算波动范围
    recent_data = historical_data.tail(5)
    
    # 计算基础下降率（每年平均下降0.5%到1.5%之间的随机值）
    base_decline_rate = np.random.uniform(-0.015, -0.005)  
    
    # 计算历史数据的波动范围
    std_dev = recent_data['consumption'].std() * 0.5  # 使用一半的标准差，使波动更加适中
    last_value = historical_data['consumption'].iloc[-1]
    
    # 初始化预测用水量
    predicted_value = last_value
    predictions = []
    
    # 为不同年份设置不同的随机种子以保证结果的多样性
    np.random.seed(sum([int(year) for year in future_years]))
    
    for i, year in enumerate(future_years):
        # 获取前一年的用水量
        if i == 0:
            base_value = last_value
        else:
            base_value = predictions[-1]
        
        # 计算当年趋势因子（基础下降率的波动）
        year_trend_rate = base_decline_rate + np.random.normal(0, 0.005)
        trend_factor = base_value * year_trend_rate
        
        # 添加随机波动
        random_factor = np.random.normal(0, 1) * std_dev
        
        # 生成当年预测值，确保整体呈下降趋势但有波动
        predicted_value = base_value + trend_factor + random_factor
        
        # 偶尔出现反弹（约25%的几率），确保整体趋势仍然是下降的
        if np.random.random() < 0.25 and i > 0:
            rebound_factor = np.random.uniform(0, 0.015) * base_value  # 反弹幅度
            predicted_value += rebound_factor
        
        # 确保预测值不会为负
        predicted_value = max(0, predicted_value)
        
        # 确保波动不会过大，限制在±5%的范围内
        max_change = 0.05 * base_value
        if abs(predicted_value - base_value) > max_change:
            if predicted_value > base_value:
                predicted_value = base_value + max_change
            else:
                predicted_value = base_value - max_change
        
        predictions.append(predicted_value)
    
    return predictions

@app.get("/")
def read_root():
    """API根路径 - 欢迎信息"""
    model_info = get_model_info()
    return {
        "message": "水资源消耗预测API服务",
        "description": "提供省级和全国水资源消耗历史数据及基于机器学习模型的预测",
        "endpoints": {
            "GET /provinces": "获取所有可用省份列表",
            "POST /predict": "根据省份ID或名称预测未来水资源消耗量",
            "GET /predict": "使用查询参数预测未来水资源消耗量"
        },
        "model": model_info
    }

@app.get("/provinces")
def get_provinces():
    """获取所有可用省份列表"""
    global data_module
    
    if data_module is None or not hasattr(data_module, "full_dataset"):
        try:
            load_models()
        except Exception as e:
            return {"error": f"加载数据模块失败: {str(e)}"}
    
    provinces = []
    try:
        # 从数据模块获取省份信息
        if data_module.full_dataset and hasattr(data_module.full_dataset, 'data'):
            # 加载合并的数据文件以获取省份ID和名称
            try:
                csv_path = os.path.join(DATA_PATH, 'merged_water_data.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    province_info = df[['province_id', 'province']].drop_duplicates()
                    
                    # 添加全国选项
                    provinces.append({
                        "id": 0,  # 使用0表示全国
                        "name": "全国"
                    })
                    
                    # 添加各省份
                    for _, row in province_info.sort_values('province_id').iterrows():
                        provinces.append({
                            "id": int(row['province_id']),
                            "name": row['province']
                        })
            except Exception as e:
                print(f"加载省份信息时出错: {e}")
                # 回退到仅获取省份名称
                for province in data_module.get_province_names():
                    provinces.append({
                        "id": None,
                        "name": province
                    })
    except Exception as e:
        return {"error": f"获取省份列表失败: {str(e)}"}
    
    return {"provinces": provinces}

# 此处删除了get_province_data函数，统一使用TimeSeriesDataModule获取省份数据

@app.post("/predict", response_model=PredictionResponse)
def predict_consumption(request: PredictionRequest):
    """预测未来水资源消耗量 (POST方法)"""
    global data_module, predictors
    
    if data_module is None:
        try:
            load_models()
        except Exception as e:
            return {"error": f"加载数据模块失败: {str(e)}"}
    
    # 确定省份ID和名称
    province_id = request.province_id
    province_name = request.province_name
    
   
    
    # 如果是要求全国数据，统一使用ID=0
    if  province_id == 0:
        province_id = None
        province_name = "全国"
    
    # 预测未来数据
    try:
        predictions = []
        
        # 根据模型类型选择预测方法
        if request.model_type == 'simulation':
            # 为模拟预测获取省份数据
            province_specific_data_module = TimeSeriesDataModule(
                data_dir=DATA_PATH,
                sequence_length=SEQUENCE_LENGTH,
                province_id=province_id
            )
            province_specific_data_module.setup()
            
            # 获取省份数据用于模拟预测
            if province_id == None:  # 全国数据
                province_data = next((df for name, df in province_specific_data_module.full_dataset.data.items() if name == "全国"), None)
            else:
                province_data = next((df for name, df in province_specific_data_module.full_dataset.data.items() 
                                if 'province_id' in df.columns and province_id in df['province_id'].values), None)
            
            if province_data is None:
                return {"error": "未找到匹配的省份数据"}
                
            # 获取省份名称
            if province_id != None:  # 非全国数据
                province_name = next((name for name, df in province_specific_data_module.full_dataset.data.items() 
                                if 'province_id' in df.columns and province_id in df['province_id'].values), "未知省份")
                
            # 使用模拟预测算法
            predictions = simulate_predictions(province_data, request.years)
            
            # 获取历史数据
            province_data = province_data.sort_values('year')
            historical_years = province_data['year'].tolist()
            historical_consumption = province_data['consumption'].tolist()
            
        else:
            # 使用机器学习模型预测
            # 检查是否有匹配的模型：优先使用特定省份的模型，如果没有则使用全国模型
            specific_key = f"{request.model_type}_{province_id}"
            global_key = f"{request.model_type}"  # 0表示全国
            
            predictor = None
            if specific_key in predictors:
                predictor = predictors[specific_key]
                print(f"使用省份专用模型: {specific_key}")
            elif global_key in predictors:
                predictor = predictors[global_key]
                print(f"使用全国通用模型: {global_key}")
            else:
                # 如果找不到匹配的模型，尝试使用任何可用的模型
                if predictors:
                    default_key = next(iter(predictors))
                    predictor = predictors[default_key]
                    print(f"找不到匹配的{request.model_type}模型，使用默认模型: {default_key}")
                else:
                    return {"error": f"没有可用的{request.model_type}预测模型"}
            
            # 为指定的省份创建专用数据模块，确保加载正确的省份数据
            province_specific_data_module = TimeSeriesDataModule(
                data_dir=DATA_PATH,
                sequence_length=SEQUENCE_LENGTH,
                province_id=province_id
            )
            
            # 确保数据已加载
            province_specific_data_module.setup()
            
            # 获取省份名称（如果只提供了省份ID）
            if province_id != None and province_name in (None, "未知"):
                for name, df in province_specific_data_module.full_dataset.data.items():
                    if 'province_id' in df.columns and province_id in df['province_id'].values:
                        province_name = name
                        break
            
            # 获取历史数据
            if province_id == None:  # 全国数据
                province_data = next((df for name, df in province_specific_data_module.full_dataset.data.items() if name == "全国"), None)
            else:
                province_data = next((df for name, df in province_specific_data_module.full_dataset.data.items() 
                                 if 'province_id' in df.columns and province_id in df['province_id'].values), None)
                
            if province_data is not None:
                province_data = province_data.sort_values('year')
                historical_years = province_data['year'].tolist()
                historical_consumption = province_data['consumption'].tolist()
            else:
                return {"error": "未找到匹配的省份数据"}
                
            # 使用特定省份的数据模块进行预测
            predictions = predictor.predict_future(
                data_module=province_specific_data_module,
                years=request.years,
                province_id=province_id,
                province_name=province_name
            )
        
        # 构建响应数据
        all_data = []
        
        # 添加历史数据
        for year, consumption in zip(historical_years, historical_consumption):
            all_data.append(DataPoint(year=year, consumption=float(consumption)))
        
        # 添加预测数据
        for year, predicted in zip(request.years, predictions):
            all_data.append(DataPoint(year=year, predicted=float(predicted)))
        
        # 获取模型信息
        model_info = get_model_info(request.model_type, province_id)
        
        # 返回结果
        return PredictionResponse(
            province_id=province_id,
            province_name=province_name,
            data=all_data,
            last_trained=model_info["last_trained"],
            model_type=model_info["model_type"]
        )
    
    except Exception as e:
        # 输出异常堆栈
        import traceback
        traceback.print_exc()
        print(f"预测失败: {e}")
        # 返回错误信息
        return {"error": f"预测失败: {str(e)}"}

@app.get("/predict", response_model=PredictionResponse)
def predict_consumption_get(
    years: List[int] = Query(default=[2024, 2025, 2026], description="需要预测的年份列表"),
    province_id: Optional[int] = Query(default=0, description="省份ID，如果不提供则预测全国数据"),
    province_name: Optional[str] = Query(default=None, description="省份名称（如果同时提供省份ID和名称，优先使用ID）"),
    model_type: str = Query(default="svm", description="预测模型类型：'svm'(支持向量机)、'rf'(随机森林)或'simulation'(模拟预测)")
):
    """预测未来水资源消耗量 (GET方法)"""
    # 创建请求对象并调用POST方法处理
    request = PredictionRequest(
        years=years,
        province_id=province_id,
        province_name=province_name,
        model_type=model_type
    )
    return predict_consumption(request)

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
