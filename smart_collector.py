#!/usr/bin/env python3
"""
智能数据收集器
一键启动数据收集，支持断点续传和自动停止
"""
import os
import sys
import asyncio
import logging
from typing import Optional
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import start_smart_collection, OpenDotaCollector
from hero_position_analyzer import HeroPositionAnalyzer
from config import load_hero_positions

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/smart_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartCollector:
    """智能数据收集器"""
    
    def __init__(self):
        self.collector = OpenDotaCollector()
        self.hero_analyzer = HeroPositionAnalyzer()
        
    def check_prerequisites(self) -> bool:
        """检查前置条件"""
        logger.info("检查前置条件...")
        
        # 检查目录
        required_dirs = ['data', 'cache', 'logs', 'outputs', 'checkpoints']
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"创建目录: {directory}")
        
        # 检查网络连接
        try:
            import aiohttp
            async def test_connection():
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://api.opendota.com/api/heroes', timeout=10) as response:
                        return response.status == 200
            
            result = asyncio.run(test_connection())
            if not result:
                logger.error("无法连接到OpenDota API")
                return False
                
        except Exception as e:
            logger.error(f"网络连接测试失败: {e}")
            return False
        
        logger.info("前置条件检查通过")
        return True
    
    async def analyze_hero_positions(self) -> bool:
        """分析英雄位置"""
        logger.info("开始分析英雄位置...")
        
        try:
            hero_positions = await self.hero_analyzer.analyze_hero_positions(sample_size=500)
            
            if hero_positions:
                logger.info(f"英雄位置分析完成，共分析 {len(hero_positions)} 个英雄")
                
                # 更新配置
                load_hero_positions()
                return True
            else:
                logger.warning("英雄位置分析失败，将使用默认配置")
                return False
                
        except Exception as e:
            logger.error(f"英雄位置分析失败: {e}")
            return False
    
    def collect_data(self, target_count: int = 20) -> Optional[dict]:
        """收集数据"""
        logger.info(f"开始收集数据，目标玩家数量: {target_count}")
        
        try:
            data = start_smart_collection(target_count)
            return data
        except Exception as e:
            logger.error(f"数据收集失败: {e}")
            return None
    
    def show_status(self):
        """显示收集状态"""
        logger.info("=== 收集状态 ===")
        
        # 检查状态文件
        state_file = os.path.join('cache', 'collector_state.json')
        if os.path.exists(state_file):
            import json
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logger.info(f"本月API使用: {state.get('monthly_request_count', 0)} / 50000")
                logger.info(f"最后请求时间: {state.get('last_request_time', 'Unknown')}")
        
        # 检查进度文件
        progress_file = os.path.join('cache', 'collection_progress.json')
        if os.path.exists(progress_file):
            import json
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                logger.info(f"已收集玩家: {len(progress.get('collected_players', []))}")
                logger.info(f"已收集比赛: {len(progress.get('collected_matches', []))}")
        
        # 检查数据文件
        data_files = [f for f in os.listdir('data') if f.startswith('team_data_')]
        logger.info(f"数据文件数量: {len(data_files)}")
        
        if data_files:
            latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join('data', x)))
            logger.info(f"最新数据文件: {latest_file}")
    
    def run_interactive(self):
        """运行交互式收集"""
        print("=== Dota2智能数据收集器 ===")
        print("功能特性:")
        print("- 自动发现玩家ID")
        print("- 断点续传")
        print("- 网络错误自动重试")
        print("- API限制智能管理")
        print("- 实时进度保存")
        print()
        
        while True:
            print("请选择操作:")
            print("1. 查看收集状态")
            print("2. 分析英雄位置")
            print("3. 开始数据收集")
            print("4. 继续上次收集")
            print("5. 退出")
            
            try:
                choice = input("\n请输入选择 (1-5): ").strip()
                
                if choice == '1':
                    self.show_status()
                    
                elif choice == '2':
                    print("开始分析英雄位置...")
                    result = asyncio.run(self.analyze_hero_positions())
                    if result:
                        print("英雄位置分析完成！")
                    else:
                        print("英雄位置分析失败，将使用默认配置")
                        
                elif choice == '3':
                    target_count = input("请输入目标玩家数量 (默认20): ").strip()
                    target_count = int(target_count) if target_count.isdigit() else 20
                    
                    print(f"开始收集 {target_count} 名玩家的数据...")
                    data = self.collect_data(target_count)
                    
                    if data:
                        print("数据收集完成！")
                        print(f"收集到 {len(data['players'])} 名玩家数据")
                        print(f"收集到 {len(data['matches'])} 场比赛数据")
                    else:
                        print("数据收集失败")
                        
                elif choice == '4':
                    print("继续上次收集...")
                    data = self.collect_data(20)
                    
                    if data:
                        print("数据收集完成！")
                    else:
                        print("数据收集失败")
                        
                elif choice == '5':
                    print("退出程序")
                    break
                    
                else:
                    print("无效选择，请输入1-5")
                    
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                logger.error(f"操作失败: {e}")
                print(f"操作失败: {e}")
            
            print("\n" + "="*50 + "\n")

def main():
    """主函数"""
    collector = SmartCollector()
    
    # 检查前置条件
    if not collector.check_prerequisites():
        print("前置条件检查失败，请检查网络连接和目录权限")
        return
    
    # 运行交互式收集
    collector.run_interactive()

if __name__ == "__main__":
    main()
