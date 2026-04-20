![Language](https://img.shields.io/badge/language-Python-blue) ![License](https://img.shields.io/badge/license-MIT-green)

# proj-Python-PCA-FaceRecognition

**基于 PCA 主成分分析的人脸识别系统，使用 Olivetti 人脸数据集训练并导出识别模型。**

## 功能特性

- PCA 降维 + 分类器人脸识别流水线
- Olivetti 标准人脸数据集支持
- 训练好的模型序列化存储（joblib）
- 可视化特征脸（Eigenface）展示
- 模块化设计，易于替换分类算法

## 快速开始

### 环境要求

- Python >= 3.8
- scikit-learn, numpy, matplotlib, joblib

### 安装步骤

```bash
git clone https://github.com/joeeei11/proj-Python-PCA-FaceRecognition.git
cd proj-Python-PCA-FaceRecognition
pip install scikit-learn numpy matplotlib joblib
```

### 基础用法

```bash
python main.py
```
