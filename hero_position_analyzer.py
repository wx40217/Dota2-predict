"""
英雄位置分析器
从OpenDota API自动获取和分析英雄的位置信息
"""
import asyncio
import aiohttp
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from collections import Counter
import numpy as np
from datetime import datetime, timedelta

from config import api_config, data_config

logger = logging.getLogger(__name__)

class HeroPositionAnalyzer:
    """英雄位置分析器"""
    
    def __init__(self):
        self.base_url = api_config.base_url
        self.timeout = api_config.timeout
        self.cache_file = os.path.join(data_config.cache_dir, "hero_positions.json")
        self.hero_stats_cache = os.path.join(data_config.cache_dir, "hero_stats.json")
        
    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """发送API请求"""
        url = f"{self.base_url}/{endpoint}"
        try:
            async with session.get(url, params=params, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API请求失败: {response.status} - {endpoint}")
                    return None
        except Exception as e:
            logger.error(f"请求异常: {e} - {endpoint}")
            return None
    
    async def get_heroes_data(self) -> List[Dict]:
        """获取所有英雄数据"""
        cache_file = os.path.join(data_config.cache_dir, "heroes.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, "heroes")
            if data:
                # 保存到缓存
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            return data or []
    
    async def get_hero_matches(self, hero_id: int, limit: int = 100) -> List[Dict]:
        """获取英雄的比赛数据"""
        async with aiohttp.ClientSession() as session:
            params = {'limit': limit}
            data = await self._make_request(session, f"heroes/{hero_id}/matches", params)
            return data or []
    
    async def get_match_details(self, match_id: int) -> Optional[Dict]:
        """获取比赛详情"""
        cache_file = os.path.join(data_config.cache_dir, f"match_{match_id}.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, f"matches/{match_id}")
            if data:
                # 保存到缓存
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            return data
    
    def analyze_hero_positions_from_matches(self, matches: List[Dict]) -> Dict[int, Dict]:
        """从比赛数据中分析英雄位置"""
        hero_positions = {}
        
        for match in matches:
            if 'players' not in match:
                continue
            
            # 按队伍分组
            radiant_players = [p for p in match['players'] if p.get('player_slot', 0) < 128]
            dire_players = [p for p in match['players'] if p.get('player_slot', 0) >= 128]
            
            # 按GPM排序确定位置（简化方法）
            for team_players in [radiant_players, dire_players]:
                # 按GPM排序
                sorted_players = sorted(team_players, 
                                      key=lambda x: x.get('gold_per_min', 0), 
                                      reverse=True)
                
                for i, player in enumerate(sorted_players):
                    hero_id = player.get('hero_id')
                    if hero_id:
                        if hero_id not in hero_positions:
                            hero_positions[hero_id] = {
                                'positions': [],
                                'gpm_stats': [],
                                'xpm_stats': [],
                                'kda_stats': [],
                                'damage_stats': [],
                                'match_count': 0
                            }
                        
                        # 记录位置（1-5号位）
                        position = i + 1
                        hero_positions[hero_id]['positions'].append(position)
                        hero_positions[hero_id]['gpm_stats'].append(player.get('gold_per_min', 0))
                        hero_positions[hero_id]['xpm_stats'].append(player.get('xp_per_min', 0))
                        hero_positions[hero_id]['kda_stats'].append(
                            (player.get('kills', 0) + player.get('assists', 0)) / max(player.get('deaths', 1), 1)
                        )
                        hero_positions[hero_id]['damage_stats'].append(player.get('hero_damage', 0))
                        hero_positions[hero_id]['match_count'] += 1
        
        return hero_positions
    
    def calculate_position_preferences(self, hero_positions: Dict[int, Dict]) -> Dict[int, Dict]:
        """计算英雄位置偏好"""
        result = {}
        
        for hero_id, stats in hero_positions.items():
            if not stats['positions']:
                continue
            
            # 计算位置分布
            position_counter = Counter(stats['positions'])
            total_matches = len(stats['positions'])
            
            # 计算每个位置的概率
            position_probs = {}
            for pos in range(1, 6):
                position_probs[f'position_{pos}'] = position_counter[pos] / total_matches
            
            # 确定主要位置
            main_position = max(position_counter, key=position_counter.get)
            
            # 计算统计信息
            avg_gpm = np.mean(stats['gpm_stats']) if stats['gpm_stats'] else 0
            avg_xpm = np.mean(stats['xpm_stats']) if stats['xpm_stats'] else 0
            avg_kda = np.mean(stats['kda_stats']) if stats['kda_stats'] else 0
            avg_damage = np.mean(stats['damage_stats']) if stats['damage_stats'] else 0
            
            result[hero_id] = {
                'main_position': main_position,
                'position_distribution': position_probs,
                'confidence': position_counter[main_position] / total_matches,
                'stats': {
                    'avg_gpm': avg_gpm,
                    'avg_xpm': avg_xpm,
                    'avg_kda': avg_kda,
                    'avg_damage': avg_damage,
                    'match_count': total_matches
                }
            }
        
        return result
    
    async def analyze_hero_positions(self, sample_size: int = 1000) -> Dict[int, Dict]:
        """分析英雄位置"""
        logger.info(f"开始分析英雄位置，样本大小: {sample_size}")
        
        # 检查缓存
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if cached_data.get('timestamp'):
                    cache_time = datetime.fromisoformat(cached_data['timestamp'])
                    if datetime.now() - cache_time < timedelta(days=7):  # 缓存7天
                        logger.info("使用缓存的英雄位置数据")
                        return cached_data['hero_positions']
        
        # 获取英雄列表
        heroes = await self.get_heroes_data()
        if not heroes:
            logger.error("无法获取英雄数据")
            return {}
        
        logger.info(f"分析 {len(heroes)} 个英雄的位置信息")
        
        all_matches = []
        hero_positions = {}
        
        # 获取每个英雄的比赛数据
        for i, hero in enumerate(heroes):
            hero_id = hero['id']
            logger.info(f"分析英雄 {hero['localized_name']} ({i+1}/{len(heroes)})")
            
            try:
                # 获取英雄比赛数据
                hero_matches = await self.get_hero_matches(hero_id, limit=50)
                
                # 获取比赛详情
                for match in hero_matches[:20]:  # 限制每个英雄20场比赛
                    match_details = await self.get_match_details(match['match_id'])
                    if match_details:
                        all_matches.append(match_details)
                
                # 如果收集到足够的数据，可以提前停止
                if len(all_matches) >= sample_size:
                    break
                    
            except Exception as e:
                logger.warning(f"分析英雄 {hero_id} 失败: {e}")
                continue
        
        # 分析位置
        logger.info(f"从 {len(all_matches)} 场比赛中分析位置信息")
        hero_positions = self.analyze_hero_positions_from_matches(all_matches)
        
        # 计算位置偏好
        result = self.calculate_position_preferences(hero_positions)
        
        # 保存结果
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(all_matches),
            'hero_positions': result
        }
        
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"英雄位置分析完成，保存到: {self.cache_file}")
        return result
    
    def get_hero_position_mapping(self) -> Dict[int, str]:
        """获取英雄位置映射"""
        if not os.path.exists(self.cache_file):
            logger.warning("英雄位置数据不存在，请先运行分析")
            return {}
        
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        hero_positions = data.get('hero_positions', {})
        mapping = {}
        
        for hero_id, info in hero_positions.items():
            main_pos = info.get('main_position', 1)
            confidence = info.get('confidence', 0)
            
            # 只保留置信度较高的映射
            if confidence > 0.6:
                mapping[int(hero_id)] = str(main_pos)
        
        return mapping
    
    def get_position_heroes(self) -> Dict[str, List[int]]:
        """按位置获取英雄列表"""
        mapping = self.get_hero_position_mapping()
        
        position_heroes = {
            '1': [],  # Carry
            '2': [],  # Mid
            '3': [],  # Offlane
            '4': [],  # Support
            '5': []   # Hard Support
        }
        
        for hero_id, position in mapping.items():
            if position in position_heroes:
                position_heroes[position].append(hero_id)
        
        return position_heroes
    
    def print_position_analysis(self):
        """打印位置分析结果"""
        if not os.path.exists(self.cache_file):
            print("英雄位置数据不存在，请先运行分析")
            return
        
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        hero_positions = data.get('hero_positions', {})
        
        print("=== 英雄位置分析结果 ===")
        print(f"分析时间: {data.get('timestamp', 'Unknown')}")
        print(f"样本大小: {data.get('sample_size', 0)} 场比赛")
        print()
        
        # 按位置分组显示
        position_heroes = self.get_position_heroes()
        
        for pos, heroes in position_heroes.items():
            position_names = {
                '1': 'Carry (1号位)',
                '2': 'Mid (2号位)', 
                '3': 'Offlane (3号位)',
                '4': 'Support (4号位)',
                '5': 'Hard Support (5号位)'
            }
            
            print(f"{position_names.get(pos, f'{pos}号位')}: {len(heroes)} 个英雄")
            
            # 显示前10个英雄
            for hero_id in heroes[:10]:
                if hero_id in hero_positions:
                    info = hero_positions[hero_id]
                    print(f"  - 英雄ID {hero_id}: 置信度 {info.get('confidence', 0):.2f}, "
                          f"比赛数 {info.get('stats', {}).get('match_count', 0)}")
            
            if len(heroes) > 10:
                print(f"  ... 还有 {len(heroes) - 10} 个英雄")
            print()

async def main():
    """主函数"""
    analyzer = HeroPositionAnalyzer()
    
    print("开始分析英雄位置...")
    hero_positions = await analyzer.analyze_hero_positions(sample_size=500)
    
    print(f"分析完成，共分析 {len(hero_positions)} 个英雄")
    
    # 打印结果
    analyzer.print_position_analysis()
    
    # 保存到配置文件
    mapping = analyzer.get_hero_position_mapping()
    print(f"生成位置映射: {len(mapping)} 个英雄")

if __name__ == "__main__":
    asyncio.run(main())
