# CS直播画面玩家血量和存活状态检测工具

这是一个用于实时检测CS直播画面中玩家血量和存活状态的Python工具。通过屏幕捕获、OCR识别和图像分析技术，自动识别游戏中每个玩家的血量数值和存活状态。

## 功能特点

- ✅ **实时屏幕捕获**: 支持指定区域的屏幕截取，降低计算开销
- ✅ **精确OCR识别**: 使用Tesseract OCR识别血量数字
- ✅ **存活状态判断**: 基于血量和图像灰度判断玩家存活状态
- ✅ **多种输出方式**: 控制台输出、JSON文件、HTTP API
- ✅ **模块化设计**: 清晰的代码结构，易于维护和扩展
- ✅ **性能优化**: 可配置检测频率，支持低CPU占用运行
- ✅ **调试支持**: 内置调试模式，可保存分析过程图像

## 快速开始

### 1. 安装依赖

运行自动安装脚本：

```bash
python install.py
```

或手动安装：

```bash
# 安装Python依赖
pip install -r requirements.txt

# Linux系统安装Tesseract
sudo apt install tesseract-ocr tesseract-ocr-eng

# Windows系统请下载Tesseract安装包
# https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. 配置检测区域

编辑 `config.json` 文件，根据你的屏幕分辨率和游戏界面调整玩家信息区域坐标：

```json
{
  "player_regions": {
    "team1": [
      {"x": 50, "y": 950, "width": 180, "height": 60, "name": "player1"},
      {"x": 250, "y": 950, "width": 180, "height": 60, "name": "player2"}
    ],
    "team2": [
      {"x": 1080, "y": 950, "width": 180, "height": 60, "name": "player1"},
      {"x": 1280, "y": 950, "width": 180, "height": 60, "name": "player2"}
    ]
  }
}
```

### 3. 运行检测

```bash
# 基本运行
python main.py

# 启用HTTP API
python main.py --api

# 调试模式（保存调试图像）
python main.py --debug --save-debug

# 自定义检测频率
python main.py --fps 10
```

## 使用方法

### 命令行参数

```bash
python main.py [选项]

选项:
  --source SOURCE    输入源 (screen/视频文件路径)
  --config CONFIG    配置文件路径 (默认: config.json)
  --debug           启用调试模式
  --save-debug      保存调试图像
  --fps FPS         检测帧率
  --api             启用HTTP API
  --port PORT       HTTP API端口
  --create-config   创建默认配置文件
```

### 输出格式

程序会输出JSON格式的检测结果：

```json
{
  "team1": [
    {"name": "Magnojez", "hp": 100, "alive": true},
    {"name": "Boombl4", "hp": 75, "alive": true},
    {"name": "Player3", "hp": 0, "alive": false}
  ],
  "team2": [
    {"name": "rain", "hp": 50, "alive": true},
    {"name": "Player2", "hp": 25, "alive": true}
  ],
  "timestamp": 1703123456789
}
```

### HTTP API

启用HTTP API后，可以通过以下端点获取数据：

- `GET /api/players` - 获取玩家信息
- `GET /api/status` - 获取服务状态
- `GET /api/config` - 获取配置信息

## 配置说明

### 主要配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `capture.source` | 输入源 (screen/视频文件) | "screen" |
| `capture.region` | 屏幕捕获区域 | 全屏 |
| `detection.fps` | 检测频率 | 5 |
| `ocr.tesseract_config` | Tesseract OCR配置 | 数字识别模式 |
| `output.console` | 控制台输出 | true |
| `output.file` | 文件输出 | true |
| `output.http_api` | HTTP API | false |

### 玩家区域配置

每个玩家区域需要配置以下参数：

- `x, y`: 区域左上角坐标
- `width, height`: 区域宽度和高度
- `name`: 玩家名称（用于输出）

建议使用调试模式来确定正确的坐标：

```bash
python main.py --debug --save-debug
```

## 性能优化

### 降低CPU使用率

1. **调整检测频率**: 将 `detection.fps` 设置为较低值（如3-5）
2. **优化捕获区域**: 只捕获包含玩家信息的屏幕区域
3. **关闭调试输出**: 生产环境中关闭 `--debug` 模式

### 提高识别准确率

1. **调整OCR配置**: 根据字体和画质调整 `tesseract_config`
2. **优化图像预处理**: 调整二值化和去噪参数
3. **精确定位血量区域**: 调整 `hp_region_offset` 参数

## 故障排除

### 常见问题

**Q: 无法识别血量数字**
A: 检查血量区域坐标是否准确，尝试调整 `hp_region_offset` 参数

**Q: 存活状态判断不准确**
A: 调整 `grayscale_threshold` 参数，或检查血量识别是否正常

**Q: 检测频率太低**
A: 增加 `detection.fps` 值，但注意CPU使用率

**Q: Tesseract OCR报错**
A: 确保正确安装Tesseract，并添加到系统PATH

### 调试方法

1. 启用调试模式查看详细日志
2. 保存调试图像检查区域提取效果
3. 查看 `cs_detector.log` 日志文件

## 系统要求

- Python 3.7+
- OpenCV 4.x
- Tesseract OCR 4.x
- 内存: 最少2GB
- CPU: 支持多核处理器

### 支持的操作系统

- Ubuntu 18.04+
- Windows 10+
- macOS 10.14+

## 项目结构

```
wjq/
├── main.py           # 主程序入口
├── capture.py        # 屏幕捕获模块
├── analyzer.py       # 图像分析和OCR模块
├── output.py         # 结果输出模块
├── config.json       # 配置文件
├── requirements.txt  # Python依赖
├── install.py        # 安装脚本
├── README.md         # 项目说明
└── debug_images/     # 调试图像目录
```

## 开发说明

### 扩展功能

1. **添加新的输出方式**: 在 `output.py` 中扩展 `OutputManager` 类
2. **改进OCR算法**: 在 `analyzer.py` 中替换或优化OCR方法
3. **支持更多游戏**: 调整 `analyzer.py` 中的游戏界面检测逻辑

### 贡献代码

1. Fork项目仓库
2. 创建功能分支
3. 提交代码更改
4. 发起Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请在GitHub上提交Issue。