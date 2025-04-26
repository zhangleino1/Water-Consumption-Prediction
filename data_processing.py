import pandas as pd
import os
import numpy as np

def clean_province_name(name):
    """移除省份名称中的省/市/自治区等后缀"""
    if isinstance(name, str):
        return name.replace('省', '').replace('市', '').replace('自治区', '') \
                   .replace('壮族', '').replace('回族', '').replace('维吾尔', '')
    return name

def process_datasets_to_csv(data_dir, output_file):
    """
    处理三个数据集并将它们合并为一个CSV文件
    
    参数:
        data_dir (str): 包含数据文件的目录路径
        output_file (str): 保存输出CSV文件的路径
    """
    # 定义文件路径
    precipitation_file = os.path.join(data_dir, '省逐年降水.xlsx')
    temperature_file = os.path.join(data_dir, '省逐年平均气温.xlsx')
    water_data_file = os.path.join(data_dir, '15-分省水资源供水和用水数据.xls')
    
    # 1. 加载并处理降水数据 (省逐年降水.xlsx)
    print("正在处理降水数据...")
    df_prec = pd.read_excel(precipitation_file)
    # 清理列名
    df_prec.rename(columns={'PR_ID': 'province_id', 'PR': 'province'}, inplace=True)
    # 清理省份名称
    df_prec['province'] = df_prec['province'].apply(clean_province_name)
    # 将数据框从宽格式转换为长格式
    year_columns = [col for col in df_prec.columns if str(col).isdigit()]
    df_prec_long = pd.melt(
        df_prec,
        id_vars=['province_id', 'province'],
        value_vars=year_columns,
        var_name='year',
        value_name='precipitation'
    )
    df_prec_long['year'] = df_prec_long['year'].astype(int)
    
    # 2. 加载并处理温度数据 (省逐年平均气温.xlsx)
    print("正在处理温度数据...")
    df_temp = pd.read_excel(temperature_file)
    # 如果存在未命名列则删除
    if 'Unnamed: 0' in df_temp.columns:
        df_temp = df_temp.drop('Unnamed: 0', axis=1)
    # 清理列名
    df_temp.rename(columns={'PR_ID': 'province_id', 'PR': 'province'}, inplace=True)
    # 清理省份名称
    df_temp['province'] = df_temp['province'].apply(clean_province_name)
    # 将数据框从宽格式转换为长格式
    year_columns = [col for col in df_temp.columns if str(col).isdigit()]
    df_temp_long = pd.melt(
        df_temp,
        id_vars=['province_id', 'province'],
        value_vars=year_columns,
        var_name='year',
        value_name='temperature'
    )
    df_temp_long['year'] = df_temp_long['year'].astype(int)
    
    # 3. 加载并处理水资源供水/消耗数据 (15-分省水资源供水和用水数据.xls)
    print("正在处理水资源供水和消耗数据...")
    df_water = pd.read_excel(water_data_file, engine='xlrd')
    
    # 打印列名用于调试
    print(f"Water data columns: {df_water.columns.tolist()}")
    
    # 根据文件中的实际列名重命名列
    column_mapping = {
        'id': 'province_id',
        'city': 'province', 
        'year': 'year',
        '水资源供水总量（亿立方米）': 'supply',
        '用水总量\n（亿立方米）': 'consumption',
        '用水总量（亿立方米）': 'consumption'  # Alternative name that might be in the file
    }
    
    # 仅重命名DataFrame中存在的列
    for old_col, new_col in column_mapping.items():
        if old_col in df_water.columns:
            df_water.rename(columns={old_col: new_col}, inplace=True)
    
    # 清理省份名称
    if 'province' in df_water.columns:
        df_water['province'] = df_water['province'].apply(clean_province_name)
    else:
        print("ERROR: 'province' column not found in water data after renaming.")
        print(f"Available columns: {df_water.columns.tolist()}")
        raise KeyError("'province' column not found in water data file")
    
    # 仅选择需要的列，确保它们存在
    required_cols = ['province', 'year', 'supply', 'consumption']
    for col in required_cols:
        if col not in df_water.columns:
            # If consumption column is missing but we have the supply column
            if col == 'consumption' and 'supply' in df_water.columns:
                print("WARNING: 'consumption' column not found, copying from 'supply'")
                df_water['consumption'] = df_water['supply']
            # If supply column is missing but we have the consumption column
            elif col == 'supply' and 'consumption' in df_water.columns:
                print("WARNING: 'supply' column not found, copying from 'consumption'")
                df_water['supply'] = df_water['consumption']
            else:
                print(f"ERROR: Required column '{col}' not found in water data.")
                print(f"Available columns: {df_water.columns.tolist()}")
                raise KeyError(f"Required column '{col}' not found in water data file")
    
    # 为最终数据集选择列
    df_water = df_water[required_cols]
    
    # 4. 合并数据集
    print("正在合并数据集...")
    # 首先合并降水和温度数据
    df_merged = pd.merge(
        df_prec_long, 
        df_temp_long, 
        on=['province', 'year', 'province_id'],
        how='inner'
    )
    
    # 然后合并供水和用水数据
    df_final = pd.merge(
        df_merged,
        df_water,
        on=['province', 'year'],
        how='inner'
    )
    
    # 5. 保存为CSV文件
    print(f"正在保存合并的数据集到 {output_file}...")
    # 使用 utf-8-sig 编码来解决中文乱码问题
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"成功创建包含 {len(df_final)} 条记录的数据集")
    
    # 返回数据样本
    return df_final.head()

if __name__ == "__main__": 
    # 设置路径
    data_dir = "dataset"
    output_file = "dataset/merged_water_data.csv"
    
    # 处理数据集并创建CSV
    sample_data = process_datasets_to_csv(data_dir, output_file)
    print("\n合并数据的样本:")
    print(sample_data)