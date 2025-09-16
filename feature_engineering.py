"""
特征工程模块
处理玩家、英雄、版本等特征，为模型训练准备数据
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

from config import data_config, HERO_POSITIONS, CURRENT_VERSION

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.hero_stats = {}
        self.version_features = {}
        
    def extract_player_features(self, player_data: Dict, matches: List[Dict]) -> Dict:
        """提取玩家特征"""
        features = {}
        
        # 基础统计特征
        if 'solo_competitive_rank' in player_data:
            features['solo_mmr'] = player_data['solo_competitive_rank']
        else:
            features['solo_mmr'] = 0
            
        if 'competitive_rank' in player_data:
            features['party_mmr'] = player_data['competitive_rank']
        else:
            features['party_mmr'] = 0
        
        # 从比赛数据计算特征
        if matches:
            recent_matches = matches[-20:]  # 最近20场比赛
            
            # 胜率
            wins = sum(1 for match in recent_matches if match.get('radiant_win', False) == (match.get('player_slot', 0) < 128))
            features['win_rate'] = wins / len(recent_matches) if recent_matches else 0
            
            # 平均KDA
            kdas = []
            for match in recent_matches:
                if 'kills' in match and 'deaths' in match and 'assists' in match:
                    deaths = max(match['deaths'], 1)  # 避免除零
                    kda = (match['kills'] + match['assists']) / deaths
                    kdas.append(kda)
            features['avg_kda'] = np.mean(kdas) if kdas else 0
            
            # 平均GPM和XPM
            gpms = [match.get('gold_per_min', 0) for match in recent_matches if 'gold_per_min' in match]
            features['avg_gpm'] = np.mean(gpms) if gpms else 0
            
            xpms = [match.get('xp_per_min', 0) for match in recent_matches if 'xp_per_min' in match]
            features['avg_xpm'] = np.mean(xpms) if xpms else 0
            
            # 英雄池大小
            heroes_played = set(match.get('hero_id') for match in recent_matches if 'hero_id' in match)
            features['hero_pool_size'] = len(heroes_played)
            
            # 位置偏好（基于英雄选择）
            position_counts = {str(i): 0 for i in range(1, 6)}
            for match in recent_matches:
                hero_id = match.get('hero_id')
                if hero_id:
                    # 根据英雄ID确定位置（这里需要实际的英雄位置映射）
                    position = self._get_hero_position(hero_id)
                    if position:
                        position_counts[position] += 1
            
            # 计算位置偏好
            total_matches = sum(position_counts.values())
            if total_matches > 0:
                for pos in position_counts:
                    features[f'position_{pos}_preference'] = position_counts[pos] / total_matches
            else:
                for pos in position_counts:
                    features[f'position_{pos}_preference'] = 0.2  # 默认平均分配
        else:
            # 默认值
            features.update({
                'win_rate': 0.5,
                'avg_kda': 1.0,
                'avg_gpm': 400,
                'avg_xpm': 500,
                'hero_pool_size': 10,
                'position_1_preference': 0.2,
                'position_2_preference': 0.2,
                'position_3_preference': 0.2,
                'position_4_preference': 0.2,
                'position_5_preference': 0.2
            })
        
        return features
    
    def _get_hero_position(self, hero_id: int) -> Optional[str]:
        """根据英雄ID获取位置"""
        try:
            from config import load_hero_positions, HERO_POSITIONS
            
            # 确保英雄位置数据已加载
            if not HERO_POSITIONS:
                load_hero_positions()
            
            # 查找英雄位置
            for position, hero_list in HERO_POSITIONS.items():
                if str(hero_id) in hero_list:
                    return position
            
            # 如果没找到，返回默认位置
            return "1"
            
        except Exception as e:
            logger.warning(f"获取英雄 {hero_id} 位置失败: {e}")
            return "1"
    
    def extract_hero_features(self, hero_data: Dict, hero_matches: List[Dict]) -> Dict:
        """提取英雄特征"""
        features = {}
        
        # 基础特征
        features['hero_id'] = hero_data.get('id', 0)
        features['localized_name'] = hero_data.get('localized_name', '')
        
        # 从比赛数据计算统计特征
        if hero_matches:
            recent_matches = hero_matches[-50:]  # 最近50场比赛
            
            # 胜率
            wins = sum(1 for match in recent_matches if match.get('radiant_win', False))
            features['win_rate'] = wins / len(recent_matches) if recent_matches else 0.5
            
            # 出场率（相对于总比赛数）
            features['pick_rate'] = len(recent_matches) / 1000  # 假设总比赛数为1000
            
            # 平均KDA
            kdas = []
            for match in recent_matches:
                if 'kills' in match and 'deaths' in match and 'assists' in match:
                    deaths = max(match['deaths'], 1)
                    kda = (match['kills'] + match['assists']) / deaths
                    kdas.append(kda)
            features['avg_kda'] = np.mean(kdas) if kdas else 1.0
            
            # 平均GPM和XPM
            gpms = [match.get('gold_per_min', 0) for match in recent_matches if 'gold_per_min' in match]
            features['avg_gpm'] = np.mean(gpms) if gpms else 400
            
            xpms = [match.get('xp_per_min', 0) for match in recent_matches if 'xp_per_min' in match]
            features['avg_xpm'] = np.mean(xpms) if xpms else 500
            
            # 平均伤害和治疗
            damages = [match.get('hero_damage', 0) for match in recent_matches if 'hero_damage' in match]
            features['avg_damage'] = np.mean(damages) if damages else 10000
            
            healings = [match.get('hero_healing', 0) for match in recent_matches if 'hero_healing' in match]
            features['avg_healing'] = np.mean(healings) if healings else 2000
        else:
            # 默认值
            features.update({
                'win_rate': 0.5,
                'pick_rate': 0.1,
                'avg_kda': 1.0,
                'avg_gpm': 400,
                'avg_xpm': 500,
                'avg_damage': 10000,
                'avg_healing': 2000
            })
        
        return features
    
    def extract_match_features(self, match_data: Dict) -> Dict:
        """提取比赛特征"""
        features = {}
        
        # 基础信息
        features['match_id'] = match_data.get('match_id', 0)
        features['duration'] = match_data.get('duration', 0)
        features['radiant_win'] = match_data.get('radiant_win', False)
        features['game_mode'] = match_data.get('game_mode', 0)
        features['lobby_type'] = match_data.get('lobby_type', 0)
        
        # 版本信息
        features['patch'] = match_data.get('patch', 0)
        features['version'] = self._extract_version(match_data.get('patch', 0))
        
        # 英雄选择
        picks_bans = match_data.get('picks_bans', [])
        if picks_bans:
            features['radiant_picks'] = [pb['hero_id'] for pb in picks_bans if pb.get('team') == 0 and pb.get('is_pick', True)]
            features['dire_picks'] = [pb['hero_id'] for pb in picks_bans if pb.get('team') == 1 and pb.get('is_pick', True)]
            features['radiant_bans'] = [pb['hero_id'] for pb in picks_bans if pb.get('team') == 0 and not pb.get('is_pick', True)]
            features['dire_bans'] = [pb['hero_id'] for pb in picks_bans if pb.get('team') == 1 and not pb.get('is_pick', True)]
        else:
            features['radiant_picks'] = []
            features['dire_picks'] = []
            features['radiant_bans'] = []
            features['dire_bans'] = []
        
        # 玩家数据
        players = match_data.get('players', [])
        if players:
            radiant_players = [p for p in players if p.get('player_slot', 0) < 128]
            dire_players = [p for p in players if p.get('player_slot', 0) >= 128]
            
            # 计算队伍平均特征
            features['radiant_avg_kda'] = self._calculate_team_avg_kda(radiant_players)
            features['dire_avg_kda'] = self._calculate_team_avg_kda(dire_players)
            features['radiant_avg_gpm'] = self._calculate_team_avg_gpm(radiant_players)
            features['dire_avg_gpm'] = self._calculate_team_avg_gpm(dire_players)
        
        return features
    
    def _extract_version(self, patch: int) -> str:
        """从patch数字提取版本号"""
        # 这里需要实际的patch到版本的映射
        # 暂时返回当前版本
        return CURRENT_VERSION
    
    def _calculate_team_avg_kda(self, players: List[Dict]) -> float:
        """计算队伍平均KDA"""
        kdas = []
        for player in players:
            kills = player.get('kills', 0)
            deaths = max(player.get('deaths', 1), 1)
            assists = player.get('assists', 0)
            kda = (kills + assists) / deaths
            kdas.append(kda)
        return np.mean(kdas) if kdas else 1.0
    
    def _calculate_team_avg_gpm(self, players: List[Dict]) -> float:
        """计算队伍平均GPM"""
        gpms = [player.get('gold_per_min', 0) for player in players if 'gold_per_min' in player]
        return np.mean(gpms) if gpms else 400
    
    def create_team_features(self, team_players: List[int], player_features: Dict, hero_features: Dict) -> Dict:
        """创建队伍特征"""
        features = {}
        
        # 队伍平均特征
        team_stats = ['win_rate', 'avg_kda', 'avg_gpm', 'avg_xpm']
        for stat in team_stats:
            values = [player_features.get(player_id, {}).get(stat, 0) for player_id in team_players]
            features[f'team_avg_{stat}'] = np.mean(values) if values else 0
            features[f'team_std_{stat}'] = np.std(values) if values else 0
        
        # 队伍英雄池多样性
        all_heroes = set()
        for player_id in team_players:
            player_data = player_features.get(player_id, {})
            hero_pool_size = player_data.get('hero_pool_size', 0)
            all_heroes.add(hero_pool_size)
        features['team_hero_diversity'] = len(all_heroes)
        
        # 位置分配合理性
        position_balance = 0
        for pos in range(1, 6):
            pos_prefs = [player_features.get(player_id, {}).get(f'position_{pos}_preference', 0.2) 
                        for player_id in team_players]
            position_balance += max(pos_prefs) - min(pos_prefs)
        features['position_balance'] = position_balance
        
        return features
    
    def prepare_training_data(self, team_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备训练数据"""
        X_bp = []  # BP特征
        X_time = []  # 时间预测特征
        X_win = []  # 胜率预测特征
        y_bp = []  # BP标签
        y_time = []  # 时间标签
        y_win = []  # 胜率标签
        
        matches = team_data.get('detailed_matches', [])
        player_features = {}
        hero_features = {}
        
        # 提取玩家特征
        for player_id, player_data in team_data.get('players', {}).items():
            player_matches = [m for m in team_data.get('matches', []) if m.get('player_slot', 0) < 128 or m.get('player_slot', 0) >= 128]
            player_features[player_id] = self.extract_player_features(player_data, player_matches)
        
        # 提取英雄特征
        for hero in team_data.get('heroes', []):
            hero_features[hero['id']] = self.extract_hero_features(hero, [])
        
        # 处理每场比赛
        for match in matches:
            match_features = self.extract_match_features(match)
            
            # BP特征和标签
            if match_features['radiant_picks'] and match_features['dire_picks']:
                bp_features = self._create_bp_features(match_features, player_features, hero_features)
                X_bp.append(bp_features)
                y_bp.append(match_features['radiant_picks'] + match_features['dire_picks'])
            
            # 时间预测特征和标签
            time_features = self._create_time_features(match_features, player_features, hero_features)
            X_time.append(time_features)
            y_time.append(match_features['duration'])
            
            # 胜率预测特征和标签
            win_features = self._create_win_features(match_features, player_features, hero_features)
            X_win.append(win_features)
            y_win.append(1 if match_features['radiant_win'] else 0)
        
        return (np.array(X_bp), np.array(X_time), np.array(X_win)), (np.array(y_bp), np.array(y_time), np.array(y_win))
    
    def _create_bp_features(self, match_features: Dict, player_features: Dict, hero_features: Dict) -> np.ndarray:
        """创建BP特征向量"""
        features = []
        
        # 队伍特征
        radiant_team = match_features.get('radiant_picks', [])
        dire_team = match_features.get('dire_picks', [])
        
        # 英雄特征
        for hero_id in radiant_team + dire_team:
            hero_data = hero_features.get(hero_id, {})
            features.extend([
                hero_data.get('win_rate', 0.5),
                hero_data.get('pick_rate', 0.1),
                hero_data.get('avg_kda', 1.0),
                hero_data.get('avg_gpm', 400),
                hero_data.get('avg_xpm', 500)
            ])
        
        # 填充到固定长度（假设最多10个英雄）
        while len(features) < 50:  # 10个英雄 * 5个特征
            features.append(0)
        
        return np.array(features[:50])
    
    def _create_time_features(self, match_features: Dict, player_features: Dict, hero_features: Dict) -> np.ndarray:
        """创建时间预测特征向量"""
        features = []
        
        # 队伍平均特征
        features.extend([
            match_features.get('radiant_avg_kda', 1.0),
            match_features.get('dire_avg_kda', 1.0),
            match_features.get('radiant_avg_gpm', 400),
            match_features.get('dire_avg_gpm', 400)
        ])
        
        # 英雄组合特征
        radiant_picks = match_features.get('radiant_picks', [])
        dire_picks = match_features.get('dire_picks', [])
        
        # 计算队伍英雄平均特征
        for team_picks in [radiant_picks, dire_picks]:
            if team_picks:
                team_hero_features = [hero_features.get(hero_id, {}) for hero_id in team_picks]
                avg_win_rate = np.mean([h.get('win_rate', 0.5) for h in team_hero_features])
                avg_gpm = np.mean([h.get('avg_gpm', 400) for h in team_hero_features])
                features.extend([avg_win_rate, avg_gpm])
            else:
                features.extend([0.5, 400])
        
        return np.array(features)
    
    def _create_win_features(self, match_features: Dict, player_features: Dict, hero_features: Dict) -> np.ndarray:
        """创建胜率预测特征向量"""
        features = []
        
        # 队伍实力对比
        features.extend([
            match_features.get('radiant_avg_kda', 1.0),
            match_features.get('dire_avg_kda', 1.0),
            match_features.get('radiant_avg_gpm', 400),
            match_features.get('dire_avg_gpm', 400)
        ])
        
        # 英雄组合强度
        radiant_picks = match_features.get('radiant_picks', [])
        dire_picks = match_features.get('dire_picks', [])
        
        for team_picks in [radiant_picks, dire_picks]:
            if team_picks:
                team_hero_features = [hero_features.get(hero_id, {}) for hero_id in team_picks]
                team_strength = np.mean([h.get('win_rate', 0.5) for h in team_hero_features])
                features.append(team_strength)
            else:
                features.append(0.5)
        
        # 版本特征
        features.append(match_features.get('patch', 0))
        
        return np.array(features)

def process_team_data(team_data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """处理团队数据文件"""
    with open(team_data_file, 'r', encoding='utf-8') as f:
        team_data = json.load(f)
    
    engineer = FeatureEngineer()
    return engineer.prepare_training_data(team_data)

if __name__ == "__main__":
    # 示例用法
    data_file = "./data/team_data_20241201_120000.json"
    if os.path.exists(data_file):
        X, y = process_team_data(data_file)
        print(f"BP特征形状: {X[0].shape}")
        print(f"时间特征形状: {X[1].shape}")
        print(f"胜率特征形状: {X[2].shape}")
        print(f"BP标签形状: {y[0].shape}")
        print(f"时间标签形状: {y[1].shape}")
        print(f"胜率标签形状: {y[2].shape}")
    else:
        print("请先运行数据收集器获取数据")
