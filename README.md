# Dota2比赛预测模型

基于OpenDota API的Dota2比赛预测系统，能够根据十人名单预测BO3系列赛的BP阵容选择、对局时间以及胜率。

## 功能特性

- **BP预测**: 预测双方的Ban/Pick阵容选择
- **时间预测**: 预测每场比赛的持续时间
- **胜率预测**: 预测双方队伍的胜率
- **位置考虑**: 考虑1-5号位的位置分配
- **版本适应**: 考虑不同版本对英雄和队伍的影响
- **BO3预测**: 完整的BO3系列赛预测

### 智能数据收集
- **自动发现玩家**: 从职业比赛和排行榜自动发现玩家ID
- **断点续传**: 支持中断后继续收集，避免重复工作
- **网络错误处理**: 自动重试和错误恢复
- **API限制管理**: 智能管理API请求频率和限制
- **实时进度保存**: 实时保存收集进度，防止数据丢失

### 英雄位置分析
- **自动位置识别**: 从比赛数据自动分析英雄位置
- **位置置信度**: 提供位置分配的置信度评估
- **动态更新**: 支持定期更新位置信息
- **多语言支持**: 支持中英文位置名称

## 系统要求

- Python 3.8+
- CUDA支持的GPU（推荐RTX 4080或更高）
- 48GB内存（推荐）
- 稳定的网络连接（用于API数据获取）

## 安装

1. 克隆项目
```bash
git clone <repository-url>
cd Dota2-predict
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 创建必要的目录
```bash
mkdir -p data cache outputs checkpoints logs
```

## 使用方法

### 1. 智能数据收集（推荐）

使用智能收集器，支持一键启动、断点续传和自动停止：

```bash
# 启动智能收集器
python smart_collector.py
```

或者使用Python接口：

```python
from data_collector import start_smart_collection

# 自动发现玩家并收集数据
data = start_smart_collection(target_count=20)
```

### 2. 手动数据收集

如果需要指定特定玩家：

```python
from data_collector import collect_data_for_players

# 指定玩家ID
player_ids = [123456789, 987654321, 111222333, 444555666, 777888999,
              101112131, 141516171, 181920212, 232425262, 272829303]

# 收集数据
data = collect_data_for_players(player_ids)
```

### 3. 英雄位置分析

自动分析英雄位置信息：

```python
from hero_position_analyzer import HeroPositionAnalyzer

analyzer = HeroPositionAnalyzer()
hero_positions = await analyzer.analyze_hero_positions()
```

### 4. 训练模型

运行完整的训练管道：

```python
from trainer import Dota2TrainingPipeline

# 创建训练管道
pipeline = Dota2TrainingPipeline()

# 运行训练
model = pipeline.run_full_pipeline(player_ids, force_refresh=False)
```

### 5. 进行预测

使用训练好的模型进行预测：

```python
from predictor import Dota2Predictor

# 创建预测器
predictor = Dota2Predictor()

# 预测BO3比赛
team_a = [123456789, 987654321, 111222333, 444555666, 777888999]
team_b = [101112131, 141516171, 181920212, 232425262, 272829303]

prediction = predictor.predict_bo3(team_a, team_b)
print(prediction)
```

## 项目结构

```
Dota2-predict/
├── config.py                    # 配置文件
├── data_collector.py            # 数据收集器
├── hero_position_analyzer.py    # 英雄位置分析器
├── feature_engineering.py       # 特征工程
├── models.py                    # 模型定义
├── trainer.py                   # 训练管道
├── predictor.py                 # 预测器
├── smart_collector.py           # 智能收集器
├── run_example.py               # 示例脚本
├── setup.py                     # 安装脚本
├── requirements.txt             # 依赖包
├── README.md                    # 说明文档
├── QUICKSTART.md                # 快速开始
├── data/                        # 数据目录
├── cache/                       # 缓存目录
├── outputs/                     # 输出目录
├── checkpoints/                 # 模型检查点
└── logs/                        # 日志目录
```

## 配置说明

### API配置
- `rate_limit_per_minute`: 每分钟请求限制（免费版60次）
- `rate_limit_per_month`: 每月请求限制（免费版50000次）
- `request_delay`: 请求间隔（秒）

### 模型配置
- `device`: 计算设备（cuda/cpu）
- `batch_size`: 批处理大小
- `learning_rate`: 学习率
- `epochs`: 训练轮数
- `early_stopping_patience`: 早停耐心值

### 数据配置
- `max_matches_per_player`: 每个玩家最大比赛数
- `recent_days`: 最近多少天的数据

## 模型架构

### BP预测模型
- 使用LSTM处理序列化的BP过程
- 考虑英雄嵌入、位置编码和版本信息
- 输出10个选择位置的英雄预测

### 时间预测模型
- 多层感知机回归模型
- 输入队伍特征和英雄组合特征
- 输出预测的比赛时间（分钟）

### 胜率预测模型
- 多层感知机分类模型
- 考虑队伍实力对比和版本因素
- 输出双方胜率概率

## 数据获取策略

考虑到OpenDota API的免费限制：

1. **分批收集**: 每次收集少量玩家的数据
2. **缓存机制**: 本地缓存已获取的数据
3. **请求限制**: 控制请求频率，避免超出限制
4. **定期更新**: 根据需要定期更新数据

## 性能优化

针对您的硬件配置（9800X3D + RTX 4080S + 48GB内存）：

1. **内存管理**: 限制最大内存使用为40GB
2. **批处理**: 使用合适的批处理大小
3. **GPU加速**: 充分利用RTX 4080S的计算能力
4. **并行处理**: 使用多线程数据加载

## 注意事项

1. **API限制**: 免费版每月50000次请求，请合理使用
2. **数据质量**: 确保玩家ID正确，数据完整
3. **版本更新**: 定期更新模型以适应新版本
4. **模型训练**: 建议使用足够的数据进行训练

## 示例输出

```json
{
  "series_info": {
    "team_a": [123456789, 987654321, 111222333, 444555666, 777888999],
    "team_b": [101112131, 141516171, 181920212, 232425262, 272829303],
    "version": "7.35"
  },
  "games": {
    "game_1": {
      "bp": {
        "radiant_picks": [1, 2, 3, 4, 5],
        "dire_picks": [6, 7, 8, 9, 10]
      },
      "duration": 35.5,
      "win_rate": {
        "team_a_win_rate": 0.55,
        "team_b_win_rate": 0.45
      }
    }
  },
  "series_prediction": {
    "predicted_series_winner": "Team A",
    "most_likely_score": "2-1",
    "series_confidence": 0.8
  }
}
```

## 故障排除

1. **内存不足**: 减少批处理大小或使用CPU训练
2. **API限制**: 增加请求间隔或使用付费API
3. **模型加载失败**: 检查模型文件路径和格式
4. **预测结果异常**: 检查输入数据格式和模型训练质量

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License
