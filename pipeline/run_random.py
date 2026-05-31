#!/usr/bin/env python3
"""
run_random.py - 执行随机选择Pipeline的便捷脚本
"""

import subprocess
import sys
from pathlib import Path

def main():
    """执行随机Pipeline"""
    
    config_file = "pipeline/random_pipeline_config.yml"
    
    if not Path(config_file).exists():
        print(f"❌ 配置文件 {config_file} 不存在")
        sys.exit(1)
    
    print("🎲 启动随机选择Pipeline...")
    print(f"使用配置文件: {config_file}")
    print("=" * 50)
    
    try:
        # 直接执行，显示实时输出
        result = subprocess.run([
            sys.executable, "pipeline/random_pipeline.py", config_file
        ], check=True)
        
        print("\n" + "=" * 50)
        print("✅ 随机Pipeline执行完成!")
        print("📊 可以使用以下命令查看结果:")
        print("   python evaluation/compare_nbv_random.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 随机Pipeline执行失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断执行")
        sys.exit(1)

if __name__ == "__main__":
    main()
