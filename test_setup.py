#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS直播画面检测工具测试脚本
验证安装和基本功能
"""

import sys
import json
import traceback
from pathlib import Path

def test_imports():
    """测试所有必要的模块导入"""
    print("测试模块导入...")
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV导入失败: {e}")
        return False
    
    try:
        import mss
        print(f"✓ MSS: {mss.__version__}")
    except ImportError as e:
        print(f"❌ MSS导入失败: {e}")
        return False
    
    try:
        import PIL
        print(f"✓ Pillow: {PIL.__version__}")
    except ImportError as e:
        print(f"❌ Pillow导入失败: {e}")
        return False
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract: {version}")
    except Exception as e:
        print(f"❌ Tesseract不可用: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    try:
        import flask
        print(f"✓ Flask: {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask导入失败: {e}")
        return False
    
    return True

def test_project_modules():
    """测试项目自定义模块"""
    print("\n测试项目模块...")
    
    try:
        from capture import create_capture_instance, ScreenCapture
        print("✓ capture模块导入成功")
    except Exception as e:
        print(f"❌ capture模块导入失败: {e}")
        return False
    
    try:
        from analyzer import PlayerAnalyzer
        print("✓ analyzer模块导入成功")
    except Exception as e:
        print(f"❌ analyzer模块导入失败: {e}")
        return False
    
    try:
        from output import OutputManager
        print("✓ output模块导入成功")
    except Exception as e:
        print(f"❌ output模块导入失败: {e}")
        return False
    
    return True

def test_config_file():
    """测试配置文件"""
    print("\n测试配置文件...")
    
    config_file = Path("config.json")
    if not config_file.exists():
        print("❌ config.json文件不存在")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必要的配置项
        required_keys = ['capture', 'player_regions', 'ocr', 'detection', 'output']
        for key in required_keys:
            if key not in config:
                print(f"❌ 配置文件缺少必要项: {key}")
                return False
        
        print("✓ 配置文件格式正确")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        # 加载配置
        with open("config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 测试屏幕捕获
        from capture import create_capture_instance
        capture = create_capture_instance(config)
        
        if hasattr(capture, 'get_screen_region'):
            frame = capture.get_screen_region()
            if frame is not None:
                print(f"✓ 屏幕捕获成功，画面尺寸: {frame.shape}")
            else:
                print("⚠ 屏幕捕获返回空值（可能是权限问题）")
        
        capture.cleanup()
        
        # 测试分析器
        from analyzer import PlayerAnalyzer
        analyzer = PlayerAnalyzer(config)
        print("✓ 分析器初始化成功")
        
        # 测试输出管理器
        from output import OutputManager
        output_manager = OutputManager(config)
        print("✓ 输出管理器初始化成功")
        
        output_manager.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("CS直播画面检测工具 - 安装验证")
    print("=" * 40)
    
    success = True
    
    # 测试模块导入
    if not test_imports():
        success = False
    
    # 测试项目模块
    if not test_project_modules():
        success = False
    
    # 测试配置文件
    if not test_config_file():
        success = False
    
    # 测试基本功能
    if not test_basic_functionality():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✅ 所有测试通过！环境配置正确。")
        print("\n可以开始使用:")
        print("python main.py")
    else:
        print("❌ 存在问题，请检查上面的错误信息。")
        print("\n建议运行:")
        print("python install.py")
    print("=" * 40)

if __name__ == "__main__":
    main()