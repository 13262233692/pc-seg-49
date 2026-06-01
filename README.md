# LiDAR 点云语义分割查看器

基于 WebGL (Three.js) 的 LiDAR 点云实时语义分割查看器，支持 LAS/LAZ 文件加载、KPConv 模型推理、手动标注修正。

## ✨ 功能特性

- **点云加载模块**
  - 支持 LAS/LAZ 格式文件上传
  - 大文件自动采样优化
  - 原始颜色与强度显示

- **推理服务模块**
  - KPConv 模型 ONNX 推理
  - 无模型时启发式分割回退
  - 支持 6 类地物识别（地面、建筑、植被、车辆、水体等）

- **渲染着色模块**
  - Three.js WebGL 高性能渲染
  - 三种渲染模式：原始颜色/语义分割/强度图
  - 可调节点大小和视角控制

- **标注修正模块**
  - 单点选择与区域选择
  - Shift 多选支持
  - 手动修改点云类别
  - 结果导出功能

## 📁 项目结构

```
pc-seg-49/
├── backend/                 # 后端服务
│   ├── app/
│   │   ├── __init__.py     # Flask 应用初始化
│   │   ├── routes.py       # API 路由
│   │   ├── pointcloud.py   # 点云处理模块
│   │   └── inference.py    # KPConv 推理模块
│   ├── models/             # ONNX 模型目录
│   ├── uploads/            # 上传文件临时目录
│   ├── requirements.txt    # Python 依赖
│   └── run.py              # 服务入口
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── components/     # React 组件
│   │   │   ├── PointCloudViewer.jsx  # 点云渲染器
│   │   │   ├── Toolbar.jsx           # 工具栏
│   │   │   ├── Sidebar.jsx           # 侧边栏
│   │   │   ├── UploadModal.jsx       # 上传界面
│   │   │   └── LoadingOverlay.jsx    # 加载动画
│   │   ├── store.js        # Zustand 状态管理
│   │   ├── utils/
│   │   │   └── api.js      # API 工具函数
│   │   ├── styles/         # 样式文件
│   │   ├── App.jsx         # 主应用组件
│   │   └── main.jsx        # 入口文件
│   ├── package.json        # Node 依赖
│   ├── vite.config.js      # Vite 配置
│   └── tailwind.config.js  # Tailwind 配置
└── start.sh                # 一键启动脚本
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+

### 一键启动

```bash
chmod +x start.sh
./start.sh
```

### 手动启动

**后端服务:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

**前端应用:**
```bash
cd frontend
npm install
npm run dev
```

### 访问应用

打开浏览器访问: http://localhost:3000

## 📖 使用指南

### 1. 上传点云文件

- 拖拽 LAS/LAZ 文件到上传区域，或点击选择文件
- 大文件会自动采样到 50000 点以优化性能

### 2. 运行语义分割

- 点击「运行语义分割」按钮
- 系统将使用 KPConv 模型（或启发式方法）进行分割
- 分割完成后自动切换到语义着色模式

### 3. 渲染模式切换

- **原始颜色**: 显示点云原始 RGB 颜色
- **语义分割**: 按类别着色显示分割结果
- **强度图**: 显示激光反射强度

### 4. 手动标注修正

1. 在侧边栏选择「选择模式」:
   - **单点**: 点击选择单个点
   - **区域**: 点击选择半径范围内的所有点

2. 按住 Shift 键可添加到现有选择

3. 在「修改类别」区域选择目标类别

4. 点击「应用类别到选中点」完成修改

### 5. 导出结果

点击「导出结果」将分割结果下载为 JSON 文件。

### 6. 变化检测模式

1. **进入变化检测模式**:
   - 完成一期点云分割后，点击「进入变化检测模式」按钮
   - 上传二期点云文件（后期）

2. **二期分割**:
   - 点击「二期语义分割」按钮
   - 等待分割完成

3. **运行变化检测**:
   - 调节「距离阈值」（默认 0.08）
   - 点击「检测建筑变化」按钮
   - 系统将自动配准两期点云并检测建筑变化

4. **查看结果**:
   - 使用「显示期数」切换查看一期/二期数据
   - 选择「变化检测」渲染模式查看变化高亮
   - 左侧边栏显示统计信息（新增/消失建筑点数、面积）

5. **变化颜色图例**:
   - 🟢 绿色 - 新增建筑
   - 🔴 红色 - 消失建筑（拆除）
   - ⚪ 灰色 - 未变化建筑
   - 🟤 棕色 - 地面
   - 🟢 深绿 - 植被

6. **导出变化结果**:
   - 点击「导出变化结果」下载 JSON 格式的变化检测报告

## 🎯 类别定义

| ID | 类别名称 | 颜色 |
|----|----------|------|
| 0 | unclassified | 灰色 |
| 1 | ground | 棕色 |
| 2 | building | 灰色 |
| 3 | vegetation | 绿色 |
| 4 | vehicle | 红色 |
| 5 | water | 蓝色 |

## 🎯 变化类型定义

| ID | 类型名称 | 颜色 | 说明 |
|----|----------|------|------|
| 0 | unchanged | 灰色 | 未变化建筑 |
| 1 | new_building | 绿色 | 新增建筑 |
| 2 | demolished | 红色 | 拆除建筑 |
| 3 | ground | 棕色 | 地面 |
| 4 | vegetation | 深绿 | 植被 |

## 🧠 模型部署

如需使用真实 KPConv ONNX 模型：

1. 将训练好的模型转换为 ONNX 格式
2. 命名为 `kpconv_model.onnx`
3. 放置在 `backend/models/` 目录下
4. 重启后端服务

模型输入形状: `(1, num_points, 6)` - [x, y, z, density, z_std, height]
模型输出形状: `(1, num_points, num_classes)`

## 🔧 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/health | 健康检查 |
| GET | /api/classes | 获取类别列表 |
| POST | /api/upload | 上传 LAS/LAZ 文件 |
| POST | /api/segment/:id | 执行语义分割 |
| POST | /api/update-labels/:id | 更新点标签 |
| GET | /api/export/:id | 导出分割结果 |

### 变化检测 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/change/types | 获取变化类型列表 |
| POST | /api/change/upload-period2 | 上传二期点云 |
| POST | /api/change/segment-period2/:id | 二期点云分割 |
| POST | /api/change/detect/:pc1_id/:pc2_id | 执行变化检测 |
| GET | /api/change/export/:change_id | 导出变化检测结果 |
| DELETE | /api/change/clear-all | 清除变化检测数据 |

## 🛠️ 技术栈

**后端:**
- Flask - Web 框架
- laspy - LAS/LAZ 文件解析
- ONNX Runtime - 模型推理
- NumPy/SciPy - 数值计算

**前端:**
- React 18 - UI 框架
- Three.js - WebGL 渲染
- Zustand - 状态管理
- Vite - 构建工具
- Tailwind CSS - 样式框架

## ⌨️ 快捷键

| 操作 | 说明 |
|------|------|
| 左键拖拽 | 旋转视角 |
| 右键拖拽 | 平移视角 |
| 滚轮 | 缩放 |
| Shift + 点击 | 添加到选择 |
