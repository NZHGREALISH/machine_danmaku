#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS直播画面检测工具安装脚本
自动安装依赖和配置环境
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import json

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        print(f"当前版本: {sys.version}")
        sys.exit(1)
    
    print(f"✓ Python版本检查通过: {sys.version}")

def install_system_dependencies():
    """安装系统依赖"""
    system = platform.system().lower()
    
    print("正在安装系统依赖...")
    
    if system == "linux":
        try:
            # 检查是否为Ubuntu/Debian系统
            if Path("/etc/apt/sources.list").exists():
                print("检测到Ubuntu/Debian系统，安装必要的系统包...")
                subprocess.run([
                    "sudo", "apt", "update"
                ], check=True)
                
                subprocess.run([
                    "sudo", "apt", "install", "-y",
                    "tesseract-ocr",
                    "tesseract-ocr-eng",
                    "tesseract-ocr-chi-sim",
                    "libgl1-mesa-glx",
                    "libglib2.0-0",
                    "libsm6",
                    "libxext6",
                    "libxrender-dev",
                    "libgomp1"
                ], check=True)
                
                print("✓ 系统依赖安装完成")
            else:
                print("⚠ 未检测到apt包管理器，请手动安装Tesseract OCR")
                print("参考命令:")
                print("  CentOS/RHEL: sudo yum install tesseract tesseract-langpack-eng")
                print("  Fedora: sudo dnf install tesseract tesseract-langpack-eng")
                print("  Arch: sudo pacman -S tesseract tesseract-data-eng")
        
        except subprocess.CalledProcessError as e:
            print(f"⚠ 系统依赖安装失败: {e}")
            print("请手动安装Tesseract OCR")
    
    elif system == "windows":
        print("Windows系统检测到，请确保已安装:")
        print("1. Tesseract OCR (https://github.com/UB-Mannheim/tesseract/wiki)")
        print("2. Microsoft Visual C++ Redistributable")
        print("3. 将Tesseract添加到系统PATH环境变量")
        
        # 检查Tesseract是否可用
        try:
            subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
            print("✓ Tesseract OCR 已安装")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ 未检测到Tesseract OCR，请手动安装")
    
    elif system == "darwin":  # macOS
        try:
            # 检查是否安装了Homebrew
            subprocess.run(["brew", "--version"], capture_output=True, check=True)
            print("检测到Homebrew，安装Tesseract...")
            
            subprocess.run(["brew", "install", "tesseract"], check=True)
            print("✓ Tesseract OCR 安装完成")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ 未检测到Homebrew，请手动安装Tesseract OCR")
            print("参考命令: brew install tesseract")

def install_python_dependencies():
    """安装Python依赖"""
    print("正在安装Python依赖包...")
    
    try:
        # 升级pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # 安装requirements.txt中的依赖
        if Path("requirements.txt").exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
        else:
            # 直接安装必要的包
            packages = [
                "opencv-python==4.9.0.80",
                "mss==9.0.1",
                "pillow==10.1.0",
                "pytesseract==0.3.10",
                "numpy==1.24.4",
                "requests==2.31.0",
                "flask==3.0.0"
            ]
            
            for package in packages:
                print(f"安装 {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        
        print("✓ Python依赖安装完成")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Python依赖安装失败: {e}")
        sys.exit(1)

def verify_installation():
    """验证安装"""
    print("正在验证安装...")
    
    required_modules = [
        "cv2",
        "mss", 
        "PIL",
        "pytesseract",
        "numpy",
        "flask"
    ]
    
    failed_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n安装失败的模块: {', '.join(failed_modules)}")
        return False
    
    # 测试Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✓ Tesseract OCR")
    except Exception as e:
        print(f"❌ Tesseract OCR: {e}")
        return False
    
    return True

def create_config_file():
    """创建配置文件"""
    if Path("config.json").exists():
        print("✓ 配置文件已存在")
        return
    
    print("正在创建默认配置文件...")
    
    # 获取屏幕分辨率来设置默认值
    try:
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # 主显示器
            width = monitor["width"]
            height = monitor["height"]
        print(f"检测到屏幕分辨率: {width}x{height}")
    except:
        width, height = 1920, 1080
        print("使用默认分辨率: 1920x1080")
    
    config = {
        "capture": {
            "source": "screen",
            "monitor": 1,
            "region": {
                "x": 0,
                "y": 0,
                "width": width,
                "height": height
            }
        },
        "player_regions": {
            "team1": [
                {"x": 50, "y": height-130, "width": 180, "height": 60, "name": "player1"},
                {"x": 250, "y": height-130, "width": 180, "height": 60, "name": "player2"},
                {"x": 450, "y": height-130, "width": 180, "height": 60, "name": "player3"},
                {"x": 650, "y": height-130, "width": 180, "height": 60, "name": "player4"},
                {"x": 850, "y": height-130, "width": 180, "height": 60, "name": "player5"}
            ],
            "team2": [
                {"x": width-850, "y": height-130, "width": 180, "height": 60, "name": "player1"},
                {"x": width-650, "y": height-130, "width": 180, "height": 60, "name": "player2"},
                {"x": width-450, "y": height-130, "width": 180, "height": 60, "name": "player3"},
                {"x": width-250, "y": height-130, "width": 180, "height": 60, "name": "player4"},
                {"x": width-50, "y": height-130, "width": 180, "height": 60, "name": "player5"}
            ]
        },
        "ocr": {
            "tesseract_config": "--psm 8 -c tessedit_char_whitelist=0123456789",
            "hp_region_offset": {
                "x": 120,
                "y": 35,
                "width": 50,
                "height": 20
            }
        },
        "detection": {
            "fps": 5,
            "alive_threshold": 0.3,
            "grayscale_threshold": 100
        },
        "output": {
            "console": True,
            "file": True,
            "file_path": "output.json",
            "http_api": False,
            "api_port": 8080
        }
    }
    
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✓ 配置文件创建完成")

def create_directories():
    """创建必要的目录"""
    directories = ["debug_images", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✓ 工作目录创建完成")

def show_usage_instructions():
    """显示使用说明"""
    print("\n" + "="*50)
    print("安装完成！")
    print("="*50)
    print("\n使用方法:")
    print("1. 基本使用:")
    print("   python main.py")
    print("")
    print("2. 启用HTTP API:")
    print("   python main.py --api")
    print("")
    print("3. 调试模式:")
    print("   python main.py --debug --save-debug")
    print("")
    print("4. 自定义检测频率:")
    print("   python main.py --fps 10")
    print("")
    print("5. 分析视频文件:")
    print("   python main.py --source video.mp4")
    print("")
    print("配置文件: config.json")
    print("日志文件: cs_detector.log")
    print("调试图片: debug_images/")
    print("")
    print("重要提示:")
    print("- 请根据实际游戏画面调整config.json中的玩家区域坐标")
    print("- 确保CS游戏运行在1920x1080分辨率下效果最佳")
    print("- 如需要HTTP API，请在防火墙中开放8080端口")

def main():
    """主安装流程"""
    print("CS直播画面玩家检测工具 - 安装脚本")
    print("="*50)
    
    try:
        # 检查Python版本
        check_python_version()
        
        # 安装系统依赖
        install_system_dependencies()
        
        # 安装Python依赖
        install_python_dependencies()
        
        # 验证安装
        if not verify_installation():
            print("\n❌ 安装验证失败，请检查错误信息")
            sys.exit(1)
        
        # 创建配置文件
        create_config_file()
        
        # 创建工作目录
        create_directories()
        
        # 显示使用说明
        show_usage_instructions()
        
    except KeyboardInterrupt:
        print("\n安装被用户中断")
    except Exception as e:
        print(f"\n安装过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()