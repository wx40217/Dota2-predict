"""
模型训练管道
整合数据收集、特征工程、模型训练和评估
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data_collector import collect_data_for_players
from feature_engineering import FeatureEngineer, process_team_data
from models import Dota2PredictionModel, ModelTrainer, Dota2Dataset, create_model
from config import model_config, data_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dota2TrainingPipeline:
    """Dota2模型训练管道"""
    
    def __init__(self):
        self.model = None
        self.trainer = None
        self.feature_engineer = FeatureEngineer()
        self.device = torch.device(model_config.device)
        
        # 创建输出目录
        os.makedirs("./outputs", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
    
    def collect_data(self, player_ids: List[int], force_refresh: bool = False) -> str:
        """收集训练数据"""
        logger.info(f"开始收集{len(player_ids)}名玩家的数据...")
        
        # 检查是否已有数据
        data_files = [f for f in os.listdir(data_config.data_dir) if f.startswith('team_data_')]
        if data_files and not force_refresh:
            latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join(data_config.data_dir, x)))
            logger.info(f"使用现有数据文件: {latest_file}")
            return os.path.join(data_config.data_dir, latest_file)
        
        # 收集新数据
        team_data = collect_data_for_players(player_ids)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_file = os.path.join(data_config.data_dir, f"team_data_{timestamp}.json")
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(team_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据收集完成，保存到: {data_file}")
        return data_file
    
    def prepare_data(self, data_file: str) -> Tuple[DataLoader, DataLoader]:
        """准备训练数据"""
        logger.info("开始准备训练数据...")
        
        # 处理数据
        X, y = process_team_data(data_file)
        X_bp, X_time, X_win = X
        y_bp, y_time, y_win = y
        
        logger.info(f"数据形状 - BP: {X_bp.shape}, 时间: {X_time.shape}, 胜率: {X_win.shape}")
        
        # 创建数据集
        dataset = Dota2Dataset(X_bp, X_time, X_win, y_bp, y_time, y_win)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.batch_size,
            shuffle=True,
            num_workers=model_config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config.batch_size,
            shuffle=False,
            num_workers=model_config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        return train_loader, val_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dota2PredictionModel:
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 创建模型
        config = {
            'device': str(self.device),
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
        
        self.model = create_model(config)
        self.trainer = ModelTrainer(self.model, config)
        
        # 训练
        self.trainer.train(train_loader, val_loader, model_config.epochs)
        
        # 加载最佳模型
        if os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))
            logger.info("已加载最佳模型权重")
        
        # 保存模型
        model_path = f"./checkpoints/dota2_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"模型已保存到: {model_path}")
        
        return self.model
    
    def evaluate_model(self, val_loader: DataLoader) -> Dict:
        """评估模型性能"""
        logger.info("开始评估模型...")
        
        self.model.eval()
        total_metrics = {'bp_accuracy': 0, 'time_mae': 0, 'win_accuracy': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(batch)
                
                # BP准确率
                bp_pred = torch.argmax(outputs['bp_predictions'], dim=-1)
                bp_acc = (bp_pred == batch['bp_labels']).float().mean()
                total_metrics['bp_accuracy'] += bp_acc.item()
                
                # 时间MAE
                time_mae = torch.abs(outputs['time_predictions'] - batch['time_labels']).mean()
                total_metrics['time_mae'] += time_mae.item()
                
                # 胜率准确率
                win_pred = (outputs['win_predictions'] > 0.5).float()
                win_acc = (win_pred == batch['win_labels']).float().mean()
                total_metrics['win_accuracy'] += win_acc.item()
        
        # 平均指标
        for key in total_metrics:
            total_metrics[key] /= len(val_loader)
        
        logger.info(f"模型评估结果:")
        logger.info(f"BP准确率: {total_metrics['bp_accuracy']:.4f}")
        logger.info(f"时间MAE: {total_metrics['time_mae']:.4f}分钟")
        logger.info(f"胜率准确率: {total_metrics['win_accuracy']:.4f}")
        
        return total_metrics
    
    def plot_training_history(self):
        """绘制训练历史"""
        if not self.trainer:
            logger.warning("没有训练历史可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # BP损失
        axes[0, 0].plot(self.trainer.train_losses['bp'], label='训练')
        axes[0, 0].plot(self.trainer.val_losses['bp'], label='验证')
        axes[0, 0].set_title('BP预测损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 时间损失
        axes[0, 1].plot(self.trainer.train_losses['time'], label='训练')
        axes[0, 1].plot(self.trainer.val_losses['time'], label='验证')
        axes[0, 1].set_title('时间预测损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 胜率损失
        axes[1, 0].plot(self.trainer.train_losses['win'], label='训练')
        axes[1, 0].plot(self.trainer.val_losses['win'], label='验证')
        axes[1, 0].set_title('胜率预测损失')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 总损失
        train_total = [sum([self.trainer.train_losses[key][i] for key in self.trainer.train_losses]) 
                      for i in range(len(self.trainer.train_losses['bp']))]
        val_total = [sum([self.trainer.val_losses[key][i] for key in self.trainer.val_losses]) 
                    for i in range(len(self.trainer.val_losses['bp']))]
        
        axes[1, 1].plot(train_total, label='训练')
        axes[1, 1].plot(val_total, label='验证')
        axes[1, 1].set_title('总损失')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('./outputs/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_bo3(self, team_a: List[int], team_b: List[int], version: str = "7.35") -> Dict:
        """预测BO3比赛"""
        if not self.model:
            raise ValueError("模型未训练，请先训练模型")
        
        logger.info(f"预测BO3比赛: 队伍A {team_a} vs 队伍B {team_b}")
        
        results = {
            'game_1': {},
            'game_2': {},
            'game_3': {},
            'series_prediction': {}
        }
        
        # 预测每场比赛
        for game in ['game_1', 'game_2', 'game_3']:
            # BP预测
            bp_result = self.model.predict_bp(team_a + team_b, version)
            
            # 时间预测
            time_result = self.model.predict_time(bp_result)
            
            # 胜率预测
            win_result = self.model.predict_win_rate(team_a, team_b, version)
            
            results[game] = {
                'bp': bp_result,
                'duration': time_result,
                'win_rate': win_result
            }
        
        # 系列赛预测
        team_a_wins = 0
        team_b_wins = 0
        
        for game in ['game_1', 'game_2', 'game_3']:
            win_rate = results[game]['win_rate']['team_a_win_rate']
            if win_rate > 0.5:
                team_a_wins += 1
            else:
                team_b_wins += 1
        
        results['series_prediction'] = {
            'team_a_wins': team_a_wins,
            'team_b_wins': team_b_wins,
            'predicted_winner': 'Team A' if team_a_wins > team_b_wins else 'Team B',
            'confidence': abs(team_a_wins - team_b_wins) / 3
        }
        
        return results
    
    def run_full_pipeline(self, player_ids: List[int], force_refresh: bool = False) -> Dota2PredictionModel:
        """运行完整训练管道"""
        logger.info("开始运行完整训练管道...")
        
        # 1. 收集数据
        data_file = self.collect_data(player_ids, force_refresh)
        
        # 2. 准备数据
        train_loader, val_loader = self.prepare_data(data_file)
        
        # 3. 训练模型
        model = self.train_model(train_loader, val_loader)
        
        # 4. 评估模型
        metrics = self.evaluate_model(val_loader)
        
        # 5. 绘制训练历史
        self.plot_training_history()
        
        # 6. 保存结果
        results = {
            'metrics': metrics,
            'config': {
                'model_config': model_config.__dict__,
                'data_config': data_config.__dict__
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('./outputs/training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("训练管道完成！")
        return model

def main():
    """主函数"""
    # 示例玩家ID列表（请替换为实际的玩家ID）
    sample_players = [
        123456789, 987654321, 111222333, 444555666, 777888999,
        101112131, 141516171, 181920212, 232425262, 272829303
    ]
    
    # 创建训练管道
    pipeline = Dota2TrainingPipeline()
    
    # 运行完整管道
    model = pipeline.run_full_pipeline(sample_players, force_refresh=False)
    
    # 示例预测
    team_a = sample_players[:5]
    team_b = sample_players[5:]
    
    prediction = pipeline.predict_bo3(team_a, team_b)
    print("\n=== BO3预测结果 ===")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
