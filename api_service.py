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
MODEL_PATH = "ml_predictions/svm_model.joblib"
DATA_PATH = "dataset"
SEQUENCE_LENGTH = 5

# 初始化数据模块和预测器
data_module = None
predictor = None

def load_models():
    global data_module, predictor
    
    # 初始化数据模块
    data_module = TimeSeriesDataModule(
        data_dir=DATA_PATH,
        sequence_length=SEQUENCE_LENGTH
    )
    
    # 确保数据已加载
    data_module.setup()
    
    # 初始化预测器
    predictor = MLWaterConsumptionPredictor(
        model_type='svm',
        sequence_length=SEQUENCE_LENGTH
    )
    
    # 尝试加载预训练的SVM模型
    try:
        predictor.load_model(MODEL_PATH)
        print(f"成功加载SVM模型: {MODEL_PATH}")
    except (FileNotFoundError, Exception) as e:
        print(f"无法加载预训练模型: {e}")
        print("将使用默认SVM模型，但预测结果可能不准确。请先训练模型。")

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    try:
        load_models()
    except Exception as e:
        print(f"启动时加载模型出错: {e}")

def get_model_info():
    """获取模型信息"""
    try:
        if os.path.exists(MODEL_PATH):
            model_data = joblib.load(MODEL_PATH)
            last_modified = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime("%Y-%m-%d %H:%M:%S")
            model_type = model_data.get('model_type', 'svm').upper()
        else:
            last_modified = "未训练"
            model_type = "SVM (未训练)"
        return {"last_trained": last_modified, "model_type": model_type}
    except Exception as e:
        print(f"获取模型信息出错: {e}")
        return {"last_trained": "未知", "model_type": "SVM"}

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

def get_province_data(province_id=None, province_name=None):
    """获取指定省份的历史数据"""
    global data_module
    
    if data_module is None or not hasattr(data_module, "full_dataset"):
        load_models()
    
    # 在数据集中查找省份数据
    province_data = None
    actual_province_name = province_name if province_name else "未知"
    actual_province_id = province_id
    
    # 如果省份ID为0或为None且省份名称为"全国"，则使用全国数据
    is_national = province_id == 0 or (province_id is None and province_name is not None and province_name == "全国")
    
    for name, df in data_module.full_dataset.data.items():
        # 全国数据
        if is_national and name == "全国":
            province_data = df
            actual_province_name = "全国"
            actual_province_id = 0
            break
        # 按ID匹配省份
        elif province_id is not None and 'province_id' in df.columns and province_id in df['province_id'].values:
            province_data = df
            actual_province_name = name
            break
        # 按名称匹配省份
        elif province_name is not None and name.lower() == province_name.lower():
            province_data = df
            actual_province_name = name
            if 'province_id' in df.columns:
                actual_province_id = df['province_id'].iloc[0]
            break
    
    # 如果未找到匹配的省份，则使用全国数据
    if province_data is None:
        for name, df in data_module.full_dataset.data.items():
            if name == "全国":
                province_data = df
                actual_province_name = "全国"
                actual_province_id = 0
                break
    
    # 如果仍然没有数据，则返回错误
    if province_data is None:
        return None, None, None
    
    return province_data, actual_province_name, actual_province_id

@app.post("/predict", response_model=PredictionResponse)
def predict_consumption(request: PredictionRequest):
    """预测未来水资源消耗量 (POST方法)"""
    global predictor, data_module
    
    if predictor is None or data_module is None:
        try:
            load_models()
        except Exception as e:
            return {"error": f"加载预测模型失败: {str(e)}"}
    
    province_data, actual_province_name, actual_province_id = get_province_data(
        province_id=request.province_id, 
        province_name=request.province_name
    )
    
    if province_data is None:
        return {"error": "未找到匹配的省份数据"}
    
    # 按年份排序
    province_data = province_data.sort_values('year')
    
    # 获取历史数据
    historical_years = province_data['year'].tolist()
    historical_consumption = province_data['consumption'].tolist()
    
    # 预测未来数据
    try:
        predictions = predictor.predict_future(
            data_module=data_module,
            years=request.years,
            province_id=actual_province_id,
            province_name=actual_province_name
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
        model_info = get_model_info()
        
        # 返回结果
        return PredictionResponse(
            province_id=actual_province_id,
            province_name=actual_province_name,
            data=all_data,
            last_trained=model_info["last_trained"],
            model_type=model_info["model_type"]
        )
    
    except Exception as e:
        return {"error": f"预测失败: {str(e)}"}

@app.get("/predict", response_model=PredictionResponse)
def predict_consumption_get(
    years: List[int] = Query(default=[2024, 2025, 2026], description="需要预测的年份列表"),
    province_id: Optional[int] = Query(default=None, description="省份ID，如果不提供则预测全国数据"),
    province_name: Optional[str] = Query(default=None, description="省份名称（如果同时提供省份ID和名称，优先使用ID）")
):
    """预测未来水资源消耗量 (GET方法)"""
    # 创建请求对象并调用POST方法处理
    request = PredictionRequest(
        years=years,
        province_id=province_id,
        province_name=province_name
    )
    return predict_consumption(request)

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
