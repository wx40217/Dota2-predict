#!/usr/bin/env python3
"""
Dota2预测模型安装和设置脚本
"""
import os
import sys
import subprocess
import logging

def setup_environment():
    """设置环境"""
    print("正在设置Dota2预测模型环境...")
    
    # 创建必要的目录
    directories = ['data', 'cache', 'outputs', 'checkpoints', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        sys.exit(1)
    
    print(f"Python版本: {sys.version}")

def install_dependencies():
    """安装依赖包"""
    print("正在安装依赖包...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("依赖包安装完成")
    except subprocess.CalledProcessError as e:
        print(f"依赖包安装失败: {e}")
        sys.exit(1)

def check_gpu():
    """检查GPU支持"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到GPU: {gpu_name} (数量: {gpu_count})")
            return True
        else:
            print("未检测到CUDA GPU，将使用CPU训练")
            return False
    except ImportError:
        print("PyTorch未安装，无法检查GPU")
        return False

def create_sample_config():
    """创建示例配置文件"""
    config_content = """# Dota2预测模型配置
# 请根据您的需求修改以下配置

[API]
rate_limit_per_minute = 60
rate_limit_per_month = 50000
request_delay = 1.0

[MODEL]
device = cuda
batch_size = 32
learning_rate = 0.001
epochs = 100
hero_embedding_dim = 128
hidden_dim = 256
num_layers = 3
dropout = 0.2

[DATA]
max_matches_per_player = 100
min_matches_per_player = 10
recent_days = 365
"""
    
    with open('config.ini', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("已创建示例配置文件: config.ini")

def main():
    """主函数"""
    print("Dota2预测模型设置向导")
    print("=" * 40)
    
    # 设置环境
    setup_environment()
    
    # 检查GPU
    has_gpu = check_gpu()
    
    # 安装依赖
    install_dependencies()
    
    # 创建配置文件
    create_sample_config()
    
    print("\n设置完成！")
    print("\n下一步:")
    print("1. 修改config.ini中的配置")
    print("2. 运行 python run_example.py 开始使用")
    print("3. 或者运行 python trainer.py 开始训练模型")
    
    if not has_gpu:
        print("\n注意: 未检测到GPU，训练速度可能较慢")
        print("建议使用GPU以获得更好的性能")

if __name__ == "__main__":
    main()
