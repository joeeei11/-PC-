# 基于神经网络的人脸识别

这是一个可以直接运行的小项目，使用神经网络 `MLPClassifier` 完成人脸身份分类识别，并提供：

- 自动下载公开演示数据集（Olivetti Faces）
- 使用你自己的图片文件夹训练模型
- 输出测试集准确率和分类报告
- 对单张人脸图片做身份预测

## 1. 环境准备

建议直接使用当前目录，在终端中执行：

```bash
python -m pip install -r requirements.txt
```

## 2. 使用公开数据集训练

下面的命令会自动下载公开数据集并训练模型：

```bash
python train.py --dataset demo
```

训练完成后，模型默认保存在：

```text
artifacts/face_recognizer.joblib
```

如果你想先把演示数据集导出到本地文件夹，方便查看图片内容，可以执行：

```bash
python prepare_demo_dataset.py
```

默认导出目录为：

```text
data/demo/olivetti_faces
```

## 3. 使用自己的数据集训练

把你的人脸图片按“一个人一个文件夹”的方式组织：

```text
data/custom/
├─ alice/
│  ├─ 001.jpg
│  ├─ 002.jpg
│  └─ 003.jpg
├─ bob/
│  ├─ 001.jpg
│  ├─ 002.jpg
│  └─ 003.jpg
```

然后运行：

```bash
python train.py --dataset custom --data-dir data/custom
```

建议：

- 每个人至少准备 5 张正脸图片
- 尽量保证光照、角度、表情有一点变化
- 图片里尽量只保留一张主要人脸

## 4. 进行预测

训练完成后，可以用模型识别人脸图片：

```bash
python predict.py --model artifacts/face_recognizer.joblib --image path/to/face.jpg
```

输出会包含：

- 预测的人名/类别
- 置信度
- 前 `k` 个最相近候选

## 5. 可调参数

你可以通过下面这些参数调整模型：

```bash
python train.py --dataset demo --hidden-layers 512,256 --pca-components 150 --max-iter 150
```

常用参数：

- `--hidden-layers`：隐藏层结构
- `--pca-components`：降维维度
- `--max-iter`：最大训练轮数
- `--learning-rate`：学习率
- `--test-size`：测试集比例

## 6. 项目说明

这套实现是“神经网络 + 图像预处理 + PCA 降维”的轻量级方案，优点是：

- 在普通 CPU 上也能快速训练
- 适合教学、课程作业和原型验证
- 代码结构简单，便于你后续改成更强的 CNN / FaceNet / ArcFace 方案

如果你后面想继续升级，我可以再帮你把它改成：

- `PyTorch` 卷积神经网络版本
- 支持摄像头实时识别
- 支持人脸检测与自动裁剪
- 支持录入你自己的成员库
