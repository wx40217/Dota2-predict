#!/usr/bin/env python3
"""
Dota2比赛预测示例脚本
演示如何使用模型进行预测
"""
import sys
import os
import logging
from typing import List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import Dota2Predictor
from trainer import Dota2TrainingPipeline

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_sample_players() -> List[int]:
    """获取示例玩家ID列表"""
    # 这些是示例ID，实际使用时请替换为真实的玩家ID
    return [
        123456789, 987654321, 111222333, 444555666, 777888999,
        101112131, 141516171, 181920212, 232425262, 272829303
    ]

def demo_prediction():
    """演示预测功能"""
    logger.info("开始演示Dota2比赛预测...")
    
    # 获取示例玩家
    all_players = get_sample_players()
    team_a = all_players[:5]
    team_b = all_players[5:]
    
    logger.info(f"队伍A: {team_a}")
    logger.info(f"队伍B: {team_b}")
    
    try:
        # 创建预测器
        logger.info("创建预测器...")
        predictor = Dota2Predictor()
        
        # 预测BO3
        logger.info("开始预测BO3比赛...")
        prediction = predictor.predict_bo3(team_a, team_b, version="7.35")
        
        # 显示结果
        print("\n" + "="*50)
        print("Dota2 BO3比赛预测结果")
        print("="*50)
        
        print(f"\n队伍A: {team_a}")
        print(f"队伍B: {team_b}")
        print(f"游戏版本: {prediction['series_info']['version']}")
        
        print(f"\n系列赛预测:")
        print(f"  预测获胜者: {prediction['series_prediction']['predicted_series_winner']}")
        print(f"  最可能比分: {prediction['series_prediction']['most_likely_score']}")
        print(f"  系列赛信心度: {prediction['series_prediction']['series_confidence']:.1%}")
        
        print(f"\n各场比赛预测:")
        for game_key, game_result in prediction['games'].items():
            print(f"\n  {game_key.upper()}:")
            print(f"    预测获胜者: {game_result['predicted_winner']}")
            print(f"    预测时长: {game_result['duration']}分钟")
            print(f"    胜率: 队伍A {game_result['win_rate']['team_a_win_rate']:.1%}, "
                  f"队伍B {game_result['win_rate']['team_b_win_rate']:.1%}")
            
            # 显示BP信息
            bp = game_result['bp']
            print(f"    BP阵容:")
            print(f"      天辉方: {bp['picks']['radiant_picks']}")
            print(f"      夜魇方: {bp['picks']['dire_picks']}")
            print(f"      天辉禁用: {bp['picks']['radiant_bans']}")
            print(f"      夜魇禁用: {bp['picks']['dire_bans']}")
        
        # 保存预测结果
        predictor.save_prediction(prediction)
        logger.info("预测结果已保存")
        
    except FileNotFoundError as e:
        logger.error(f"模型文件未找到: {e}")
        print("\n错误: 未找到训练好的模型文件")
        print("请先运行训练管道训练模型:")
        print("python trainer.py")
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        print(f"\n预测失败: {e}")

def demo_training():
    """演示训练功能"""
    logger.info("开始演示模型训练...")
    
    # 获取示例玩家
    all_players = get_sample_players()
    
    try:
        # 创建训练管道
        logger.info("创建训练管道...")
        pipeline = Dota2TrainingPipeline()
        
        # 运行训练（注意：这会消耗API请求次数）
        logger.info("开始训练模型...")
        logger.warning("注意: 训练过程会消耗OpenDota API请求次数")
        
        # 询问用户是否继续
        response = input("是否继续训练？这将消耗API请求次数 (y/N): ")
        if response.lower() != 'y':
            logger.info("用户取消训练")
            return
        
        model = pipeline.run_full_pipeline(all_players, force_refresh=False)
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        print(f"\n训练失败: {e}")

def main():
    """主函数"""
    print("Dota2比赛预测系统")
    print("="*30)
    print("1. 演示预测功能")
    print("2. 演示训练功能")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请选择功能 (1-3): ").strip()
            
            if choice == '1':
                demo_prediction()
                break
            elif choice == '2':
                demo_training()
                break
            elif choice == '3':
                print("退出程序")
                break
            else:
                print("无效选择，请输入1-3")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            logger.error(f"程序异常: {e}")
            print(f"程序异常: {e}")
            break

if __name__ == "__main__":
    main()
