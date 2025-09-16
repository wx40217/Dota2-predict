# 快速开始指南

## 1. 环境设置

```bash
# 运行设置脚本
python setup.py

# 或者手动安装依赖
pip install -r requirements.txt
```

## 2. 配置修改

编辑 `config.py` 文件，根据您的需求调整配置：

- **API限制**: 免费版每月50000次请求
- **硬件配置**: 适配您的9800X3D + RTX 4080S + 48GB内存
- **数据设置**: 调整每个玩家的比赛数量等

## 3. 数据收集

```python
from data_collector import collect_data_for_players

# 替换为实际的玩家ID
player_ids = [123456789, 987654321, 111222333, 444555666, 777888999,
              101112131, 141516171, 181920212, 232425262, 272829303]

# 收集数据（注意API限制）
data = collect_data_for_players(player_ids)
```

## 4. 模型训练

```python
from trainer import Dota2TrainingPipeline

# 创建训练管道
pipeline = Dota2TrainingPipeline()

# 运行训练
model = pipeline.run_full_pipeline(player_ids, force_refresh=False)
```

## 5. 进行预测

```python
from predictor import Dota2Predictor

# 创建预测器
predictor = Dota2Predictor()

# 预测BO3
team_a = [123456789, 987654321, 111222333, 444555666, 777888999]
team_b = [101112131, 141516171, 181920212, 232425262, 272829303]

prediction = predictor.predict_bo3(team_a, team_b)
print(prediction)
```

## 6. 运行示例

```bash
# 运行交互式示例
python run_example.py
```

## 重要提醒

1. **API限制**: 免费版每月50000次请求，请合理使用
2. **数据质量**: 确保玩家ID正确，建议使用知名职业选手ID
3. **训练时间**: 首次训练可能需要较长时间，请耐心等待
4. **版本更新**: 定期更新模型以适应新版本

## 故障排除

### 内存不足
- 减少 `batch_size` 到 16 或 8
- 减少 `max_matches_per_player` 到 50

### API限制
- 增加 `request_delay` 到 2.0 秒
- 分批收集数据

### GPU问题
- 检查CUDA安装
- 设置 `device = "cpu"` 使用CPU训练

## 获取帮助

如果遇到问题，请检查：
1. Python版本 >= 3.8
2. 依赖包是否正确安装
3. 网络连接是否正常
4. 玩家ID是否有效
