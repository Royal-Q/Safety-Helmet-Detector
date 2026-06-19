# 安全帽佩戴检测系统 使用说明

基于 YOLOv5 构建的图形化安全帽佩戴检测工具，支持图片、视频和摄像头实时检测，并具备违规告警、截图保存、日志记录等功能。

---

## 1. 环境要求

- **操作系统**：Windows / Linux / macOS
- **Python 版本**：**3.10**（推荐）

---

## 2. 安装依赖


```bash
pip install -r requirements.txt
```

如果下载缓慢，可更换国内镜像源，例如：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```


## 3. 使用方法

### 3.1 使用命令行脚本进行单张图片测试（可选）

项目包含 `test.py`，用于从 `test_images` 文件夹读取第一张图片进行检测，结果保存到 `test_results` 并自动显示。

在项目根目录下运行：

```bash
python test.py
```



### 3.2 启动 GUI 主程序

在项目根目录下运行：

```bash
python gui.py
```

程序启动后，会自动加载模型（状态栏显示“模型已加载”后方可使用）。
![主界面](images/detection_result.jpg)

#### 主界面布局：
- **左侧面板**：模型状态、检测源按钮（图片/视频/摄像头）、参数调节、检测统计、告警记录。
- **右侧区域**：显示检测画面。
- **底部**：进度条、控制按钮（暂停/停止/重新播放）、保存按钮。

#### 操作步骤：
1. **选择检测源**：
   - 点击 **“图片检测”**：选择单张图片，检测完成后显示结果并可保存。
   - 点击 **“视频检测”**：选择视频文件，开始逐帧检测，支持暂停/继续、停止、重新播放，并可保存检测后的视频。
   - 点击 **“摄像头实时检测”**：打开默认摄像头进行实时检测，点击同一按钮可关闭摄像头。

2. **调节参数**（可选）：
   - 置信度阈值（0.1~0.9）：数值越高，检测框越严格。
   - IoU 阈值（0.1~0.9）：用于非极大值抑制，数值越高重叠框被抑制越多。

3. **查看统计**：
   - **图片模式**：显示当前帧检测到的人数（佩戴/未佩戴）。
   - **视频/摄像头模式**：采用简易目标跟踪，显示整个过程中**不同人员**的累计数量（去重），有效避免重复计数。

4. **告警记录**：
   - 当检测到未佩戴安全帽时，告警信息会出现在左侧列表（时间戳、置信度），并自动保存违规截图至 `alerts/screenshots/`，同时在 `alerts/alert_log.txt` 中记录日志。
   - **仅摄像头模式**会弹出告警弹窗（3秒内防抖），图片/视频模式只记录不弹窗。

5. **保存结果**：
   - 图片检测完成后，点击 **“保存”** 可将标注后的图片保存为 JPG/PNG。
   - 视频检测完成后，点击 **“保存”** 可将标注后的完整视频保存为 MP4/AVI。


---

### **4. 模型训练**


#### **4.1 数据集**

本项目的训练代码使用 YOLO 格式的数据集。


公开数据集 [**SHWD (Safety Helmet Wearing Dataset)**](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) 提供了完整的标注数据。你可以通过以下链接获取已转换为 YOLO 格式的数据集：
- **百度网盘链接**：https://pan.baidu.com/s/1CceCFIYzpBjjPcCe4_dr7g
- **提取码**：`gyre`

下载并解压后，数据集的目录结构应如下所示：

```
dataset/
├── images/
│   ├── train/      # 训练集图片
│   ├── val/        # 验证集图片
│   └── test/       # 测试集图片
└── labels/
    ├── train/      # 训练集标注
    ├── val/        # 验证集标注
    └── test/       # 测试集标注
```


#### **4.2 配置训练环境**


首先，确保已安装项目依赖：

```bash
pip install -r requirements.txt
```

#### **4.3 创建数据集配置文件**

在项目根目录下创建 `helmet_data.yaml` 文件，配置训练和验证数据的路径：

```yaml
train: /path/to/dataset/images/train
val:   /path/to/dataset/images/val
test:  /path/to/dataset/images/test  

# 类别数量
nc: 2

# 类别名称
names: ['no_helmet', 'helmet']
```

> **注意**：请将 `/path/to/dataset/` 替换为你的数据集实际存放路径。

#### **4.4 修改模型配置文件**

在 `models/` 目录下，选择 `yolov5s.yaml`，将其中的类别数 `nc` 修改为 `2`。

```yaml
# 找到这一行并修改
nc: 2  # 类别数量
```

#### **4.5 训练权重**

为了提高训练效率和最终模型的精度，建议使用在 COCO 数据集上预训练的权重 `yolov5s.pt` 作为起点。你可以从官方渠道下载：

```bash
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
```

#### **4.6 执行训练**

在项目根目录下运行以下命令开始训练：

```bash
python train.py --epochs 200 \
                --data helmet_data.yaml \
                --cfg models/yolov5s.yaml \
                --weights yolov5s.pt \
                --batch-size 64 \
                --device 0 \
                --workers 8
```

**参数说明**：

| 参数 | 说明 |
| :--- | :--- |
| `--epochs` | 训练轮数，200轮通常可以达到较好收敛 |
| `--data` | 数据集配置文件的路径 |
| `--cfg` | 模型结构配置文件的路径 |
| `--weights` | 预训练权重文件的路径 |
| `--batch-size` | 批大小，可根据 GPU 显存调整 |
| `--device` | 训练设备，`0` 表示使用第一张 GPU，`cpu` 表示使用 CPU |
| `--workers` | 数据加载的线程数 |

#### **4.7 训练结果**

训练完成后，所有结果会保存在 `runs/train/exp/` 目录下：

| 文件/目录 | 说明 |
| :--- | :--- |
| `weights/best.pt` | 验证集上表现最佳的模型权重 |
| `weights/last.pt` | 最后一个 epoch 的模型权重 |
| `results.png` | 训练损失及精度/mAP曲线图 |
| `PR_curve.png` | 各类别的精确率-召回率曲线 |
| `confusion_matrix.png` | 分类结果混淆矩阵 |
| `results.csv` | 每个 epoch 的详细指标数据 |


#### **4.8 训练效果参考**

使用 SHWD 数据集（训练集 5457 张，验证集 607 张，测试集 1517 张），训练 200 轮，最终模型在测试集上的性能为：
- **mAP@0.5**：94.7%
- **精确率（Precision）**：93.3%
- **召回率（Recall）**：90.5%

![训练曲线](runs\train\exp1\results.png)

## 5. 目录与文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `gui.py` | 主 GUI 程序 |
| `test.py` | 单张图片快速检测脚本 |
| `requirements.txt` | Python 依赖清单 |
| `models/` | YOLOv5 模型定义 |
| `utils/` | YOLOv5 工具函数 |
| `runs/train/exp1/weights/best.pt` | 默认权重路径 |
| `test_images/` | 存放待检测图片 |
| `test_results/` | 脚本输出结果图片 |
| `alerts/alert_log.txt` | 告警日志 |
| `alerts/screenshots/` | 违规截图 |

---