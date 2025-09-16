"""
Dota2比赛预测模型
包含BP预测、时间预测和胜率预测三个子模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from config import model_config

logger = logging.getLogger(__name__)

class Dota2Dataset(Dataset):
    """Dota2数据集类"""
    
    def __init__(self, X_bp, X_time, X_win, y_bp, y_time, y_win):
        self.X_bp = torch.FloatTensor(X_bp)
        self.X_time = torch.FloatTensor(X_time)
        self.X_win = torch.FloatTensor(X_win)
        self.y_bp = torch.LongTensor(y_bp)
        self.y_time = torch.FloatTensor(y_time)
        self.y_win = torch.FloatTensor(y_win)
    
    def __len__(self):
        return len(self.X_bp)
    
    def __getitem__(self, idx):
        return {
            'bp_features': self.X_bp[idx],
            'time_features': self.X_time[idx],
            'win_features': self.X_win[idx],
            'bp_labels': self.y_bp[idx],
            'time_labels': self.y_time[idx],
            'win_labels': self.y_win[idx]
        }

class HeroEmbedding(nn.Module):
    """英雄嵌入层"""
    
    def __init__(self, num_heroes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_heroes, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, hero_ids):
        return self.embedding(hero_ids)

class PositionEncoder(nn.Module):
    """位置编码器，处理1-5号位信息"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.position_embedding = nn.Embedding(6, embedding_dim)  # 0-5号位
        self.embedding_dim = embedding_dim
    
    def forward(self, positions):
        return self.position_embedding(positions)

class VersionEncoder(nn.Module):
    """版本编码器，处理版本信息"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.version_embedding = nn.Embedding(100, embedding_dim)  # 支持100个版本
        self.embedding_dim = embedding_dim
    
    def forward(self, versions):
        return self.version_embedding(versions)

class BPPredictionModel(nn.Module):
    """BP预测模型"""
    
    def __init__(self, input_dim: int, hero_embedding_dim: int, hidden_dim: int, 
                 num_heroes: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.hero_embedding = HeroEmbedding(num_heroes, hero_embedding_dim)
        self.position_encoder = PositionEncoder(hero_embedding_dim)
        self.version_encoder = VersionEncoder(hero_embedding_dim)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim + hero_embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM层用于序列建模
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 输出层
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, num_heroes) for _ in range(10)  # 10个选择位置
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, hero_ids, positions, versions):
        # 英雄嵌入
        hero_emb = self.hero_embedding(hero_ids)
        pos_emb = self.position_encoder(positions)
        ver_emb = self.version_encoder(versions)
        
        # 特征融合
        combined_features = torch.cat([features, hero_emb, pos_emb, ver_emb], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # LSTM处理
        lstm_out, _ = self.lstm(fused_features.unsqueeze(1))
        lstm_out = self.dropout(lstm_out.squeeze(1))
        
        # 输出预测
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(lstm_out))
        
        return torch.stack(outputs, dim=1)  # [batch_size, 10, num_heroes]

class TimePredictionModel(nn.Module):
    """比赛时间预测模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

class WinPredictionModel(nn.Module):
    """胜率预测模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

class Dota2PredictionModel(nn.Module):
    """Dota2综合预测模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 模型参数
        input_dim = config.get('input_dim', 50)
        hero_embedding_dim = config.get('hero_embedding_dim', 128)
        hidden_dim = config.get('hidden_dim', 256)
        num_heroes = config.get('num_heroes', 120)
        num_layers = config.get('num_layers', 3)
        dropout = config.get('dropout', 0.2)
        
        # 子模型
        self.bp_model = BPPredictionModel(
            input_dim, hero_embedding_dim, hidden_dim, 
            num_heroes, num_layers, dropout
        )
        
        self.time_model = TimePredictionModel(
            input_dim, hidden_dim, num_layers, dropout
        )
        
        self.win_model = WinPredictionModel(
            input_dim, hidden_dim, num_layers, dropout
        )
        
        # 移动到设备
        self.to(self.device)
    
    def forward(self, batch):
        """前向传播"""
        bp_features = batch['bp_features'].to(self.device)
        time_features = batch['time_features'].to(self.device)
        win_features = batch['win_features'].to(self.device)
        
        # 模拟输入（实际使用时需要根据数据调整）
        hero_ids = torch.randint(0, 120, (bp_features.size(0), 10)).to(self.device)
        positions = torch.randint(1, 6, (bp_features.size(0), 10)).to(self.device)
        versions = torch.randint(0, 100, (bp_features.size(0), 1)).to(self.device)
        
        # 各子模型预测
        bp_predictions = self.bp_model(bp_features, hero_ids, positions, versions)
        time_predictions = self.time_model(time_features)
        win_predictions = self.win_model(win_features)
        
        return {
            'bp_predictions': bp_predictions,
            'time_predictions': time_predictions,
            'win_predictions': win_predictions
        }
    
    def predict_bp(self, player_list: List[int], version: str = "7.35") -> Dict:
        """预测BP阵容"""
        self.eval()
        with torch.no_grad():
            # 这里需要根据实际输入格式调整
            # 暂时返回示例结果
            return {
                'radiant_picks': [1, 2, 3, 4, 5],
                'dire_picks': [6, 7, 8, 9, 10],
                'radiant_bans': [11, 12, 13, 14, 15],
                'dire_bans': [16, 17, 18, 19, 20],
                'pick_order': ['radiant_pick', 'dire_pick', 'radiant_pick', 'dire_pick', 'radiant_pick']
            }
    
    def predict_time(self, team_composition: Dict) -> float:
        """预测比赛时间（分钟）"""
        self.eval()
        with torch.no_grad():
            # 这里需要根据实际输入格式调整
            # 暂时返回示例结果
            return 35.5
    
    def predict_win_rate(self, team_a: List[int], team_b: List[int], version: str = "7.35") -> Dict:
        """预测胜率"""
        self.eval()
        with torch.no_grad():
            # 这里需要根据实际输入格式调整
            # 暂时返回示例结果
            return {
                'team_a_win_rate': 0.55,
                'team_b_win_rate': 0.45,
                'confidence': 0.8
            }

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model: Dota2PredictionModel, config: Dict):
        self.model = model
        self.config = config
        self.device = model.device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 损失函数
        self.bp_criterion = nn.CrossEntropyLoss()
        self.time_criterion = nn.MSELoss()
        self.win_criterion = nn.BCELoss()
        
        # 训练历史
        self.train_losses = {'bp': [], 'time': [], 'win': []}
        self.val_losses = {'bp': [], 'time': [], 'win': []}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_losses = {'bp': 0, 'time': 0, 'win': 0}
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(batch)
            
            # 计算损失
            bp_loss = self.bp_criterion(
                outputs['bp_predictions'].view(-1, outputs['bp_predictions'].size(-1)),
                batch['bp_labels'].view(-1)
            )
            
            time_loss = self.time_criterion(
                outputs['time_predictions'],
                batch['time_labels']
            )
            
            win_loss = self.win_criterion(
                outputs['win_predictions'],
                batch['win_labels']
            )
            
            # 总损失
            total_loss = bp_loss + time_loss + win_loss
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失
            total_losses['bp'] += bp_loss.item()
            total_losses['time'] += time_loss.item()
            total_losses['win'] += win_loss.item()
        
        # 平均损失
        for key in total_losses:
            total_losses[key] /= len(dataloader)
            self.train_losses[key].append(total_losses[key])
        
        return total_losses
    
    def validate(self, dataloader: DataLoader) -> Dict:
        """验证模型"""
        self.model.eval()
        total_losses = {'bp': 0, 'time': 0, 'win': 0}
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(batch)
                
                # 计算损失
                bp_loss = self.bp_criterion(
                    outputs['bp_predictions'].view(-1, outputs['bp_predictions'].size(-1)),
                    batch['bp_labels'].view(-1)
                )
                
                time_loss = self.time_criterion(
                    outputs['time_predictions'],
                    batch['time_labels']
                )
                
                win_loss = self.win_criterion(
                    outputs['win_predictions'],
                    batch['win_labels']
                )
                
                total_losses['bp'] += bp_loss.item()
                total_losses['time'] += time_loss.item()
                total_losses['win'] += win_loss.item()
        
        # 平均损失
        for key in total_losses:
            total_losses[key] /= len(dataloader)
            self.val_losses[key].append(total_losses[key])
        
        return total_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        """训练模型"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_losses['bp'] + val_losses['time'] + val_losses['win'])
            
            # 早停
            current_val_loss = val_losses['bp'] + val_losses['time'] + val_losses['win']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.get('early_stopping_patience', 10):
                logger.info(f"早停于epoch {epoch}")
                break
            
            # 打印进度
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: "
                          f"Train Loss - BP: {train_losses['bp']:.4f}, "
                          f"Time: {train_losses['time']:.4f}, "
                          f"Win: {train_losses['win']:.4f}")
                logger.info(f"Val Loss - BP: {val_losses['bp']:.4f}, "
                          f"Time: {val_losses['time']:.4f}, "
                          f"Win: {val_losses['win']:.4f}")

def create_model(config: Dict = None) -> Dota2PredictionModel:
    """创建模型实例"""
    if config is None:
        config = {
            'device': model_config.device,
            'input_dim': 50,
            'hero_embedding_dim': model_config.hero_embedding_dim,
            'hidden_dim': model_config.hidden_dim,
            'num_heroes': 120,
            'num_layers': model_config.num_layers,
            'dropout': model_config.dropout,
            'learning_rate': model_config.learning_rate,
            'weight_decay': model_config.weight_decay,
            'early_stopping_patience': model_config.early_stopping_patience
        }
    
    return Dota2PredictionModel(config)

if __name__ == "__main__":
    # 示例用法
    model = create_model()
    print(f"模型已创建，设备: {model.device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试预测
    sample_players = [123456789, 987654321, 111222333, 444555666, 777888999,
                     101112131, 141516171, 181920212, 232425262, 272829303]
    
    bp_result = model.predict_bp(sample_players)
    time_result = model.predict_time({})
    win_result = model.predict_win_rate(sample_players[:5], sample_players[5:])
    
    print(f"BP预测结果: {bp_result}")
    print(f"时间预测结果: {time_result}分钟")
    print(f"胜率预测结果: {win_result}")
