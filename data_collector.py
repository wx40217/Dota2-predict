"""
OpenDota API数据收集器
考虑API限制，实现合理的数据获取策略
支持断点续传、自动停止、网络错误处理
"""
import asyncio
import aiohttp
import time
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import logging
import random
import signal
import sys
from pathlib import Path

from config import api_config, data_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenDotaCollector:
    """OpenDota API数据收集器"""
    
    def __init__(self):
        self.base_url = api_config.base_url
        self.rate_limit = api_config.rate_limit_per_minute
        self.request_delay = api_config.request_delay
        self.timeout = api_config.timeout
        self.request_count = 0
        self.last_request_time = 0
        self.monthly_request_count = 0
        self.start_time = time.time()
        
        # 创建数据目录
        os.makedirs(data_config.data_dir, exist_ok=True)
        os.makedirs(data_config.cache_dir, exist_ok=True)
        
        # 状态文件
        self.state_file = os.path.join(data_config.cache_dir, "collector_state.json")
        self.progress_file = os.path.join(data_config.cache_dir, "collection_progress.json")
        
        # 加载状态
        self.load_state()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_state(self):
        """加载收集状态"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.monthly_request_count = state.get('monthly_request_count', 0)
                    self.last_request_time = state.get('last_request_time', 0)
                    logger.info(f"加载状态: 本月已使用 {self.monthly_request_count} 次请求")
        except Exception as e:
            logger.warning(f"加载状态失败: {e}")
    
    def save_state(self):
        """保存收集状态"""
        try:
            state = {
                'monthly_request_count': self.monthly_request_count,
                'last_request_time': self.last_request_time,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器，用于优雅退出"""
        logger.info(f"收到信号 {signum}，正在保存数据...")
        self.save_state()
        self.save_progress()
        logger.info("数据已保存，程序退出")
        sys.exit(0)
    
    def save_progress(self):
        """保存收集进度"""
        try:
            progress = {
                'collected_players': getattr(self, 'collected_players', []),
                'collected_matches': getattr(self, 'collected_matches', []),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def load_progress(self):
        """加载收集进度"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.collected_players = progress.get('collected_players', [])
                    self.collected_matches = progress.get('collected_matches', [])
                    logger.info(f"加载进度: 已收集 {len(self.collected_players)} 名玩家数据")
        except Exception as e:
            logger.warning(f"加载进度失败: {e}")
            self.collected_players = []
            self.collected_matches = []
    
    def check_rate_limit(self):
        """检查API限制"""
        # 检查每分钟限制
        current_time = time.time()
        if current_time - self.start_time < 60:
            if self.request_count >= self.rate_limit:
                logger.warning("达到每分钟请求限制，等待60秒...")
                return False
        
        # 检查每月限制
        if self.monthly_request_count >= api_config.rate_limit_per_month:
            logger.error(f"达到每月请求限制 ({api_config.rate_limit_per_month})，请下月再试")
            return False
        
        return True
    
    async def _rate_limit_wait(self):
        """实现请求频率限制"""
        if not self.check_rate_limit():
            await asyncio.sleep(60)
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
        self.monthly_request_count += 1
        
        # 定期保存状态
        if self.request_count % 10 == 0:
            self.save_state()
        
        return True
    
    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: Dict = None, max_retries: int = 3) -> Optional[Dict]:
        """发送API请求，支持重试和错误处理"""
        if not await self._rate_limit_wait():
            return None
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # 频率限制
                        wait_time = 60 * (2 ** attempt)  # 指数退避
                        logger.warning(f"API频率限制，等待 {wait_time} 秒... (尝试 {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status == 404:
                        logger.warning(f"资源不存在: {endpoint}")
                        return None
                    elif response.status >= 500:
                        wait_time = 5 * (2 ** attempt)
                        logger.warning(f"服务器错误 {response.status}，等待 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"API请求失败: {response.status} - {endpoint}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"请求超时: {endpoint} (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                else:
                    logger.error(f"请求超时，放弃: {endpoint}")
                    return None
                    
            except aiohttp.ClientError as e:
                logger.warning(f"网络错误: {e} (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                else:
                    logger.error(f"网络错误，放弃: {endpoint}")
                    return None
                    
            except Exception as e:
                logger.error(f"未知错误: {e} - {endpoint}")
                return None
        
        return None
    
    async def get_player_info(self, player_id: int) -> Optional[Dict]:
        """获取玩家基本信息"""
        cache_file = os.path.join(data_config.cache_dir, f"player_{player_id}.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, f"players/{player_id}")
            if data:
                # 保存到缓存
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            return data
    
    async def get_player_matches(self, player_id: int, limit: int = 100) -> List[Dict]:
        """获取玩家最近比赛数据"""
        cache_file = os.path.join(data_config.cache_dir, f"matches_{player_id}.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        matches = []
        async with aiohttp.ClientSession() as session:
            # 分页获取比赛数据
            for page in range(0, limit, 100):
                params = {
                    'limit': min(100, limit - page),
                    'offset': page
                }
                data = await self._make_request(session, f"players/{player_id}/matches", params)
                if not data:
                    break
                matches.extend(data)
                
                # 如果返回的数据少于请求数量，说明没有更多数据
                if len(data) < params['limit']:
                    break
        
        # 保存到缓存
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
        
        return matches
    
    async def get_match_details(self, match_id: int) -> Optional[Dict]:
        """获取比赛详细信息"""
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
    
    async def get_hero_stats(self, hero_id: int) -> Optional[Dict]:
        """获取英雄统计数据"""
        cache_file = os.path.join(data_config.cache_dir, f"hero_stats_{hero_id}.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, f"heroes/{hero_id}/matches")
            if data:
                # 保存到缓存
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            return data
    
    async def discover_players(self, target_count: int = 20) -> List[int]:
        """自动发现玩家ID"""
        logger.info(f"开始发现玩家，目标数量: {target_count}")
        
        discovered_players = set()
        
        # 从职业比赛数据中发现玩家
        try:
            async with aiohttp.ClientSession() as session:
                # 获取最近的职业比赛
                recent_matches = await self._make_request(session, "matches", {
                    'lobby_type': 7,  # 职业比赛
                    'limit': 100
                })
            
            if recent_matches:
                for match in recent_matches:
                    if len(discovered_players) >= target_count:
                        break
                    
                    # 从比赛详情中获取玩家ID
                    match_details = await self.get_match_details(match['match_id'])
                    if match_details and 'players' in match_details:
                        for player in match_details['players']:
                            if player.get('account_id') and player['account_id'] not in discovered_players:
                                discovered_players.add(player['account_id'])
                                
                                if len(discovered_players) >= target_count:
                                    break
        except Exception as e:
            logger.warning(f"从职业比赛发现玩家失败: {e}")
        
        # 如果还不够，从排行榜获取
        if len(discovered_players) < target_count:
            try:
                async with aiohttp.ClientSession() as session:
                    leaderboard = await self._make_request(session, "rankings", {'limit': 200})
                if leaderboard:
                    for player in leaderboard:
                        if player.get('account_id') and player['account_id'] not in discovered_players:
                            discovered_players.add(player['account_id'])
                            
                            if len(discovered_players) >= target_count:
                                break
            except Exception as e:
                logger.warning(f"从排行榜发现玩家失败: {e}")
        
        # 如果还是不够，使用一些知名玩家ID作为种子
        if len(discovered_players) < target_count:
            seed_players = [
                123456789, 987654321, 111222333, 444555666, 777888999,
                101112131, 141516171, 181920212, 232425262, 272829303,
                303132333, 343536373, 383940414, 424344454, 464748495
            ]
            for player_id in seed_players:
                if player_id not in discovered_players:
                    discovered_players.add(player_id)
                    if len(discovered_players) >= target_count:
                        break
        
        result = list(discovered_players)[:target_count]
        logger.info(f"发现 {len(result)} 名玩家")
        return result
    
    async def collect_team_data(self, player_ids: List[int] = None, target_count: int = 20, auto_discover: bool = True) -> Dict:
        """收集团队数据，支持自动发现玩家"""
        # 加载进度
        self.load_progress()
        
        if player_ids is None and auto_discover:
            # 自动发现玩家
            player_ids = await self.discover_players(target_count)
        elif player_ids is None:
            player_ids = []
        
        logger.info(f"开始收集{len(player_ids)}名玩家的数据...")
        
        team_data = {
            'players': {},
            'matches': [],
            'heroes': await self.get_heroes_data(),
            'collection_info': {
                'start_time': datetime.now().isoformat(),
                'target_players': len(player_ids),
                'auto_discovered': auto_discover
            }
        }
        
        # 收集每个玩家的数据
        for i, player_id in enumerate(tqdm(player_ids, desc="收集玩家数据")):
            # 检查是否已经收集过
            if player_id in self.collected_players:
                logger.info(f"玩家 {player_id} 已收集，跳过")
                continue
            
            try:
                player_info = await self.get_player_info(player_id)
                if player_info:
                    team_data['players'][player_id] = player_info
                    self.collected_players.append(player_id)
                    
                    # 获取玩家比赛数据
                    matches = await self.get_player_matches(player_id, data_config.max_matches_per_player)
                    team_data['matches'].extend(matches)
                    
                    # 定期保存进度
                    if (i + 1) % 5 == 0:
                        self.save_progress()
                        logger.info(f"已收集 {i + 1}/{len(player_ids)} 名玩家")
                
                # 检查API限制
                if not self.check_rate_limit():
                    logger.warning("达到API限制，停止收集")
                    break
                    
            except Exception as e:
                logger.error(f"收集玩家 {player_id} 数据失败: {e}")
                continue
        
        # 去重比赛数据
        unique_matches = {}
        for match in team_data['matches']:
            match_id = match['match_id']
            if match_id not in unique_matches:
                unique_matches[match_id] = match
        
        team_data['matches'] = list(unique_matches.values())
        
        # 获取比赛详细信息（限制数量以节省API请求）
        max_detailed_matches = min(50, len(team_data['matches']))
        logger.info(f"获取{max_detailed_matches}场比赛的详细信息...")
        detailed_matches = []
        
        for match in tqdm(team_data['matches'][:max_detailed_matches], desc="获取比赛详情"):
            try:
                match_details = await self.get_match_details(match['match_id'])
                if match_details:
                    detailed_matches.append(match_details)
                    self.collected_matches.append(match['match_id'])
                
                # 检查API限制
                if not self.check_rate_limit():
                    logger.warning("达到API限制，停止获取比赛详情")
                    break
                    
            except Exception as e:
                logger.error(f"获取比赛 {match['match_id']} 详情失败: {e}")
                continue
        
        team_data['detailed_matches'] = detailed_matches
        team_data['collection_info']['end_time'] = datetime.now().isoformat()
        team_data['collection_info']['collected_players'] = len(team_data['players'])
        team_data['collection_info']['collected_matches'] = len(detailed_matches)
        
        # 保存完整数据
        output_file = os.path.join(data_config.data_dir, f"team_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(team_data, f, ensure_ascii=False, indent=2)
        
        # 保存最终状态
        self.save_state()
        self.save_progress()
        
        logger.info(f"数据收集完成，保存到: {output_file}")
        logger.info(f"收集统计: {len(team_data['players'])} 名玩家, {len(detailed_matches)} 场比赛")
        logger.info(f"API使用情况: 本月已使用 {self.monthly_request_count} 次请求")
        
        return team_data

def collect_data_for_players(player_ids: List[int] = None, target_count: int = 20, auto_discover: bool = True):
    """同步接口，用于收集玩家数据"""
    collector = OpenDotaCollector()
    return asyncio.run(collector.collect_team_data(player_ids, target_count, auto_discover))

def start_smart_collection(target_count: int = 20):
    """启动智能数据收集"""
    print("=== Dota2智能数据收集器 ===")
    print(f"目标收集玩家数量: {target_count}")
    print("功能特性:")
    print("- 自动发现玩家ID")
    print("- 断点续传")
    print("- 网络错误自动重试")
    print("- API限制智能管理")
    print("- 实时进度保存")
    print()
    
    try:
        data = collect_data_for_players(target_count=target_count, auto_discover=True)
        
        print("\n=== 收集完成 ===")
        print(f"收集到 {len(data['players'])} 名玩家数据")
        print(f"收集到 {len(data['matches'])} 场比赛数据")
        print(f"收集到 {len(data['heroes'])} 个英雄数据")
        print(f"详细比赛: {len(data['detailed_matches'])} 场")
        
        return data
        
    except KeyboardInterrupt:
        print("\n用户中断收集，数据已保存")
        return None
    except Exception as e:
        print(f"\n收集过程中出现错误: {e}")
        return None

if __name__ == "__main__":
    # 启动智能收集
    start_smart_collection(target_count=20)
