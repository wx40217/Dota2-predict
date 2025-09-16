"""
Dota2比赛预测器
提供简单的接口进行比赛预测
"""
import torch
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

from models import Dota2PredictionModel, create_model
from config import model_config

logger = logging.getLogger(__name__)

class Dota2Predictor:
    """Dota2比赛预测器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device(model_config.device)
        
        if model_path:
            self.load_model(model_path)
        else:
            # 尝试加载最新的模型
            self.load_latest_model()
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        try:
            config = {
                'device': str(self.device),
                'input_dim': 50,
                'hero_embedding_dim': model_config.hero_embedding_dim,
                'hidden_dim': model_config.hidden_dim,
                'num_heroes': 120,
                'num_layers': model_config.num_layers,
                'dropout': model_config.dropout
            }
            
            self.model = create_model(config)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"模型已加载: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def load_latest_model(self):
        """加载最新的模型"""
        import os
        import glob
        
        # 查找最新的模型文件
        model_files = glob.glob("./checkpoints/dota2_model_*.pth")
        if not model_files:
            raise FileNotFoundError("未找到训练好的模型文件")
        
        latest_model = max(model_files, key=os.path.getctime)
        self.load_model(latest_model)
    
    def predict_bp(self, player_list: List[int], version: str = "7.35") -> Dict:
        """
        预测BP阵容
        
        Args:
            player_list: 10名玩家的ID列表
            version: 游戏版本
            
        Returns:
            BP预测结果
        """
        if not self.model:
            raise ValueError("模型未加载")
        
        if len(player_list) != 10:
            raise ValueError("需要提供10名玩家")
        
        # 模拟BP预测（实际实现需要根据训练数据调整）
        team_a = player_list[:5]
        team_b = player_list[5:]
        
        # 这里应该使用实际的模型预测
        # 暂时返回示例结果
        bp_result = {
            'radiant_team': team_a,
            'dire_team': team_b,
            'picks': {
                'radiant_picks': [1, 2, 3, 4, 5],  # 英雄ID
                'dire_picks': [6, 7, 8, 9, 10],
                'radiant_bans': [11, 12, 13, 14, 15],
                'dire_bans': [16, 17, 18, 19, 20]
            },
            'pick_order': [
                'radiant_pick_1', 'dire_pick_1', 'dire_pick_2',
                'radiant_pick_2', 'radiant_pick_3', 'dire_pick_3',
                'radiant_ban_1', 'dire_ban_1', 'dire_ban_2',
                'radiant_ban_2', 'radiant_ban_3', 'dire_ban_3',
                'dire_pick_4', 'radiant_pick_4', 'radiant_pick_5',
                'dire_pick_5', 'radiant_ban_4', 'dire_ban_4',
                'dire_ban_5', 'radiant_ban_5'
            ],
            'strategy_analysis': {
                'radiant_strategy': '团战控制流',
                'dire_strategy': '分推带线流',
                'key_heroes': ['英雄A', '英雄B', '英雄C'],
                'counter_picks': ['克制英雄1', '克制英雄2']
            }
        }
        
        return bp_result
    
    def predict_duration(self, team_composition: Dict) -> float:
        """
        预测比赛时间
        
        Args:
            team_composition: 队伍阵容信息
            
        Returns:
            预测的比赛时间（分钟）
        """
        if not self.model:
            raise ValueError("模型未加载")
        
        # 这里应该使用实际的模型预测
        # 暂时返回示例结果
        base_time = 35.0
        
        # 根据阵容调整时间
        if 'strategy_analysis' in team_composition:
            strategy = team_composition['strategy_analysis']
            if '团战' in strategy.get('radiant_strategy', ''):
                base_time += 5.0  # 团战流通常时间更长
            elif '分推' in strategy.get('dire_strategy', ''):
                base_time -= 3.0  # 分推流可能结束更快
        
        return round(base_time, 1)
    
    def predict_win_rate(self, team_a: List[int], team_b: List[int], version: str = "7.35") -> Dict:
        """
        预测胜率
        
        Args:
            team_a: 队伍A的玩家ID列表
            team_b: 队伍B的玩家ID列表
            version: 游戏版本
            
        Returns:
            胜率预测结果
        """
        if not self.model:
            raise ValueError("模型未加载")
        
        if len(team_a) != 5 or len(team_b) != 5:
            raise ValueError("每支队伍需要5名玩家")
        
        # 这里应该使用实际的模型预测
        # 暂时返回示例结果
        team_a_strength = 0.55  # 队伍A实力
        team_b_strength = 0.45  # 队伍B实力
        
        # 添加一些随机性
        import random
        noise = random.uniform(-0.05, 0.05)
        team_a_win_rate = max(0.1, min(0.9, team_a_strength + noise))
        team_b_win_rate = 1 - team_a_win_rate
        
        confidence = abs(team_a_win_rate - team_b_win_rate) * 2
        
        return {
            'team_a_win_rate': round(team_a_win_rate, 3),
            'team_b_win_rate': round(team_b_win_rate, 3),
            'confidence': round(confidence, 3),
            'predicted_winner': 'Team A' if team_a_win_rate > team_b_win_rate else 'Team B',
            'analysis': {
                'team_a_advantages': ['选手实力强', '英雄池深', '配合默契'],
                'team_b_advantages': ['版本理解好', '战术灵活', '经验丰富'],
                'key_factors': ['BP策略', '选手状态', '版本适应']
            }
        }
    
    def predict_bo3(self, team_a: List[int], team_b: List[int], version: str = "7.35") -> Dict:
        """
        预测BO3系列赛
        
        Args:
            team_a: 队伍A的玩家ID列表
            team_b: 队伍B的玩家ID列表
            version: 游戏版本
            
        Returns:
            BO3预测结果
        """
        if len(team_a) != 5 or len(team_b) != 5:
            raise ValueError("每支队伍需要5名玩家")
        
        logger.info(f"开始预测BO3: 队伍A {team_a} vs 队伍B {team_b}")
        
        results = {
            'series_info': {
                'team_a': team_a,
                'team_b': team_b,
                'version': version,
                'prediction_time': datetime.now().isoformat()
            },
            'games': {},
            'series_prediction': {}
        }
        
        # 预测每场比赛
        for game_num in range(1, 4):
            game_key = f'game_{game_num}'
            
            # 模拟每场比赛的BP
            all_players = team_a + team_b
            bp_result = self.predict_bp(all_players, version)
            
            # 预测比赛时间
            duration = self.predict_duration(bp_result)
            
            # 预测胜率
            win_rate = self.predict_win_rate(team_a, team_b, version)
            
            results['games'][game_key] = {
                'bp': bp_result,
                'duration': duration,
                'win_rate': win_rate,
                'predicted_winner': win_rate['predicted_winner']
            }
        
        # 系列赛预测
        team_a_wins = 0
        team_b_wins = 0
        
        for game_key, game_result in results['games'].items():
            if game_result['win_rate']['predicted_winner'] == 'Team A':
                team_a_wins += 1
            else:
                team_b_wins += 1
        
        # 计算系列赛胜率
        series_win_rate = self.predict_win_rate(team_a, team_b, version)
        
        results['series_prediction'] = {
            'team_a_wins': team_a_wins,
            'team_b_wins': team_b_wins,
            'predicted_series_winner': 'Team A' if team_a_wins > team_b_wins else 'Team B',
            'series_confidence': series_win_rate['confidence'],
            'most_likely_score': f"{max(team_a_wins, team_b_wins)}-{min(team_a_wins, team_b_wins)}",
            'analysis': {
                'key_games': ['Game 1', 'Game 2'] if team_a_wins + team_b_wins < 3 else ['Game 1', 'Game 2', 'Game 3'],
                'decisive_factors': ['BP策略', '选手状态', '版本适应', '团队配合'],
                'recommendations': {
                    'team_a': ['加强团战配合', '准备多套阵容', '注意版本强势英雄'],
                    'team_b': ['提高个人操作', '优化BP策略', '加强沟通协调']
                }
            }
        }
        
        return results
    
    def save_prediction(self, prediction: Dict, filename: str = None):
        """保存预测结果"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"./outputs/prediction_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(prediction, f, ensure_ascii=False, indent=2)
        
        logger.info(f"预测结果已保存到: {filename}")

def main():
    """主函数 - 示例用法"""
    # 示例玩家ID
    team_a = [123456789, 987654321, 111222333, 444555666, 777888999]
    team_b = [101112131, 141516171, 181920212, 232425262, 272829303]
    
    try:
        # 创建预测器
        predictor = Dota2Predictor()
        
        # 预测BO3
        prediction = predictor.predict_bo3(team_a, team_b)
        
        # 打印结果
        print("=== Dota2 BO3预测结果 ===")
        print(f"队伍A: {team_a}")
        print(f"队伍B: {team_b}")
        print(f"预测获胜者: {prediction['series_prediction']['predicted_series_winner']}")
        print(f"最可能比分: {prediction['series_prediction']['most_likely_score']}")
        print(f"系列赛信心度: {prediction['series_prediction']['series_confidence']}")
        
        print("\n=== 各场比赛预测 ===")
        for game_key, game_result in prediction['games'].items():
            print(f"\n{game_key.upper()}:")
            print(f"  预测获胜者: {game_result['predicted_winner']}")
            print(f"  预测时长: {game_result['duration']}分钟")
            print(f"  胜率: 队伍A {game_result['win_rate']['team_a_win_rate']:.1%}, "
                  f"队伍B {game_result['win_rate']['team_b_win_rate']:.1%}")
        
        # 保存预测结果
        predictor.save_prediction(prediction)
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        print(f"预测失败: {e}")

if __name__ == "__main__":
    main()
