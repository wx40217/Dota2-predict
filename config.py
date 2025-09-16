"""
Dota2比赛预测模型配置文件
"""
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """OpenDota API配置"""
    base_url: str = "https://api.opendota.com/api"
    rate_limit_per_minute: int = 60  # 免费版每分钟60次请求
    rate_limit_per_month: int = 50000  # 免费版每月50000次请求
    request_delay: float = 1.0  # 请求间隔（秒）
    timeout: int = 30  # 请求超时时间

@dataclass
class ModelConfig:
    """模型配置"""
    # 硬件配置
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    max_memory_gb: int = 40  # 最大内存使用（GB），留8GB给系统
    batch_size: int = 32  # 批处理大小
    num_workers: int = 8  # 数据加载线程数
    
    # 模型参数
    hero_embedding_dim: int = 128  # 英雄嵌入维度
    player_embedding_dim: int = 64  # 玩家嵌入维度
    hidden_dim: int = 256  # 隐藏层维度
    num_layers: int = 3  # 网络层数
    dropout: float = 0.2  # Dropout率
    
    # 训练参数
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # 位置配置
    positions: List[str] = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = ["1", "2", "3", "4", "5"]  # 1-5号位

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    max_matches_per_player: int = 100  # 每个玩家最大比赛数
    min_matches_per_player: int = 10   # 每个玩家最小比赛数
    recent_days: int = 365  # 最近多少天的数据
    
    # 特征工程
    hero_features: List[str] = None
    player_features: List[str] = None
    
    def __post_init__(self):
        if self.hero_features is None:
            self.hero_features = [
                "win_rate", "pick_rate", "ban_rate", "avg_kda", 
                "avg_gpm", "avg_xpm", "avg_damage", "avg_healing"
            ]
        
        if self.player_features is None:
            self.player_features = [
                "win_rate", "avg_kda", "avg_gpm", "avg_xpm", 
                "preferred_roles", "hero_pool_size", "recent_performance"
            ]

# 全局配置实例
api_config = APIConfig()
model_config = ModelConfig()
data_config = DataConfig()

# 英雄位置映射（将从API自动获取）
HERO_POSITIONS = {}

def load_hero_positions():
    """加载英雄位置映射"""
    global HERO_POSITIONS
    try:
        from hero_position_analyzer import HeroPositionAnalyzer
        analyzer = HeroPositionAnalyzer()
        position_heroes = analyzer.get_position_heroes()
        
        # 转换为字符串ID列表格式
        HERO_POSITIONS = {
            '1': [str(hero_id) for hero_id in position_heroes['1']],
            '2': [str(hero_id) for hero_id in position_heroes['2']],
            '3': [str(hero_id) for hero_id in position_heroes['3']],
            '4': [str(hero_id) for hero_id in position_heroes['4']],
            '5': [str(hero_id) for hero_id in position_heroes['5']]
        }
        
        logger.info(f"加载英雄位置映射: {sum(len(heroes) for heroes in HERO_POSITIONS.values())} 个英雄")
        return True
    except Exception as e:
        logger.warning(f"加载英雄位置映射失败: {e}")
        # 使用默认映射
        HERO_POSITIONS = {
            "1": ["1", "2", "3", "4", "5"],  # 示例ID
            "2": ["6", "7", "8", "9", "10"],
            "3": ["11", "12", "13", "14", "15"],
            "4": ["16", "17", "18", "19", "20"],
            "5": ["21", "22", "23", "24", "25"]
        }
        return False

# 位置名称映射
POSITION_NAMES = {
    "1": "Carry (1号位)",
    "2": "Mid (2号位)", 
    "3": "Offlane (3号位)",
    "4": "Support (4号位)",
    "5": "Hard Support (5号位)"
}

# 位置英文名称
POSITION_EN_NAMES = {
    "1": "Carry",
    "2": "Mid", 
    "3": "Offlane",
    "4": "Support",
    "5": "Hard Support"
}

# 版本信息
CURRENT_VERSION = "7.35"  # 当前版本，需要定期更新
VERSION_FEATURES = [
    "hero_balance_changes", "item_changes", "meta_shift", 
    "new_heroes", "removed_heroes"
]
