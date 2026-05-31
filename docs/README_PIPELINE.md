# 3DGS Uncertainty 完整Pipeline系统

## 📋 系统概述

本系统实现了3D Gaussian Splatting的不确定性评估和Next Best View (NBV)选择，支持NBV方法与随机选择方法的对比分析。

## 🗂️ 核心文件结构

### NBV Pipeline (智能视角选择)
- `unified_pipeline.py` - NBV主流程执行脚本
- `pipeline_config.yml` - NBV配置文件

### Random Pipeline (随机基线对比) 
- `random_pipeline.py` - 随机选择主流程脚本
- `random_pipeline_config.yml` - 随机选择配置文件

### 结果分析工具
- `calculate_metrics.py` - 单次结果计算
- `summary_metrics.py` - 多数据集结果汇总
- `combined_metrics.py` - 综合结果分析
- `compare_nbv_random.py` - NBV vs Random对比分析

### 便捷执行脚本
- `run_random.py` - 一键执行随机Pipeline

## 🚀 使用方法

### 1. 执行NBV Pipeline
```bash
python unified_pipeline.py pipeline_config.yml
```

### 2. 执行Random Pipeline
```bash
python random_pipeline.py random_pipeline_config.yml
# 或使用便捷脚本
python run_random.py
```

### 3. 对比分析结果
```bash
python compare_nbv_random.py
```

## ⚙️ 技术规格

### 训练参数
- **总迭代数**: 30,000
- **patch_size**: 4 (不确定性计算)
- **NBV触发点**: 16个 (每1875次迭代)
- **保存检查点**: 7,000, 30,000

### NBV流程
1. 生成初始4个视角 (`gen_split.py`)
2. 训练基础模型
3. 渲染所有视角计算不确定性
4. 基于Fisher信息选择最佳视角
5. 添加新视角继续训练
6. 重复步骤3-5直到完成

### Random流程
1. 生成初始4个视角
2. 训练基础模型
3. 随机选择新视角添加
4. 继续训练
5. 重复步骤3-4直到完成

## 📊 评估指标

- **PSNR** (Peak Signal-to-Noise Ratio) - 越高越好
- **SSIM** (Structural Similarity Index) - 越高越好  
- **LPIPS** (Learned Perceptual Image Patch Similarity) - 越低越好

## 📁 输出结构

### NBV结果
```
results/
├── kitchen/
│   ├── point_cloud/
│   ├── renders/
│   └── results.json
├── counter/
└── bonsai/
```

### Random结果
```
random_results/
├── kitchen/
│   ├── point_cloud/
│   ├── renders/
│   └── results.json
├── counter/
└── bonsai/
```

## 🔍 已验证结果 (NBV方法)

| 数据集  | PSNR   | SSIM   | LPIPS  |
|---------|--------|--------|--------|
| kitchen | 21.2438| 0.7621 | 0.2002 |
| counter | 21.1846| 0.7186 | 0.2362 |
| bonsai  | 19.2255| 0.6791 | 0.2791 |

## 🎯 关键特性

✅ **多数据集批处理** - 自动处理配置中的所有数据集  
✅ **结果持久化** - 自动保存metrics到JSON文件  
✅ **4位小数精度** - 满足精确度要求  
✅ **错误恢复** - 单个数据集失败不影响后续处理  
✅ **实时日志** - 详细的执行进度显示  
✅ **对比分析** - NBV vs Random完整对比  

## 🔧 故障排除

### 常见问题
1. **CUDA内存不足** - 减少batch_size或使用更小的patch_size
2. **数据集路径错误** - 确认data/目录下有对应的数据集文件夹
3. **依赖包缺失** - 检查requirements.txt中的包是否都已安装

### 调试技巧
- 使用`--debug`参数获取更详细的日志
- 检查临时文件夹权限
- 监控GPU内存使用情况

## 📈 下一步计划

1. **执行Random Pipeline** - 获取基线对比结果
2. **详细对比分析** - 量化NBV方法的改进效果
3. **可视化结果** - 生成对比图表
4. **参数优化** - 基于对比结果调整NBV策略

---

*最后更新: 2025-01-11*  
*系统状态: NBV Pipeline已验证，Random Pipeline已创建待执行*
