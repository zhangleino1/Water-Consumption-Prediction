# 水资源消耗预测API服务使用指南

本文档介绍如何使用水资源消耗预测API服务，该服务基于SVM模型，可以预测中国各省份和全国的水资源消耗情况。

## 前提条件

确保已经安装所有必要的依赖：

```bash
pip install fastapi uvicorn scikit-learn pandas numpy matplotlib pytorch-lightning torch tensorboard
```

## 步骤1: 训练SVM模型

在启动API服务之前，我们需要先训练一个SVM模型。使用提供的脚本进行训练：

```bash
python train_svm_model.py --tune_hyperparameters
```

这将训练一个全国范围的SVM模型。如果您想针对特定省份训练模型，可以添加`--province_id`参数：

```bash
# 以河南省(410000)为例训练模型
python train_svm_model.py --province_id 410000 --tune_hyperparameters
```

## 步骤2: 启动API服务

训练完成后，启动FastAPI服务：

```bash
python api_service.py
```

服务将在 http://localhost:8000 运行。

## 步骤3: 使用API

API提供了以下几个端点：

### 1. 根端点（健康检查）

- **URL**: `/`
- **方法**: `GET`
- **描述**: 返回API概览信息和模型状态

```bash
curl http://localhost:8000/
```

### 2. 获取省份列表

- **URL**: `/provinces`
- **方法**: `GET`
- **描述**: 返回所有可用省份及其ID

```bash
curl http://localhost:8000/provinces
```

### 3. 预测水资源消耗 (GET方法)

- **URL**: `/predict`
- **方法**: `GET`
- **参数**:
  - `years`: 预测年份列表 (默认: 2024,2025,2026)
  - `province_id`: 省份ID (可选)
  - `province_name`: 省份名称 (可选)

```bash
# 预测河南省未来5年水资源消耗
curl "http://localhost:8000/predict?years=2024&years=2025&years=2026&years=2027&years=2028&province_id=410000"

# 预测全国未来3年水资源消耗
curl "http://localhost:8000/predict?years=2024&years=2025&years=2026"
```

### 4. 预测水资源消耗 (POST方法)

- **URL**: `/predict`
- **方法**: `POST`
- **请求体**: JSON格式

```bash
# 预测河南省未来5年水资源消耗
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"years":[2024,2025,2026,2027,2028],"province_id":410000}'

# 使用省份名称而非ID预测
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"years":[2024,2025,2026],"province_name":"河南"}'

# 预测全国数据
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"years":[2024,2025,2026]}'
```

## 响应示例

```json
{
  "province_id": 410000,
  "province_name": "河南",
  "data": [
    {"year": 2000, "consumption": 123.45, "predicted": null},
    {"year": 2001, "consumption": 125.67, "predicted": null},
    // ... 历史数据 ...
    {"year": 2023, "consumption": 145.32, "predicted": null},
    {"year": 2024, "consumption": null, "predicted": 146.78},
    {"year": 2025, "consumption": null, "predicted": 147.93},
    {"year": 2026, "consumption": null, "predicted": 148.54}
  ],
  "last_trained": "2025-05-10 15:30:22",
  "model_type": "SVM"
}
```

## 在网页浏览器中访问API文档

FastAPI自动生成了交互式API文档，可以在浏览器中访问：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

通过这些界面，您可以直接在浏览器中测试API端点。
