#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS直播画面玩家血量和存活状态检测工具
主程序入口

使用方法:
    python main.py --source screen              # 从屏幕捕获
    python main.py --source video.mp4          # 从视频文件分析
    python main.py --config custom_config.json # 使用自定义配置
    python main.py --debug                     # 启用调试模式
"""

import argparse
import json
import logging
import time
import signal
import sys
import os
from pathlib import Path
import cv2

# 导入自定义模块
from capture import create_capture_instance
from analyzer import PlayerAnalyzer, create_debug_image
from output import OutputManager

class CSPlayerDetector:
    """CS玩家检测器主类"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化检测器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.running = False
        
        # 初始化组件
        self.capture = None
        self.analyzer = None
        self.output_manager = None
        
        # 调试选项
        self.debug_mode = False
        self.save_debug_images = False
        
        # 性能统计
        self.frame_count = 0
        self.start_time = None
        
        self.setup_logging()
        self.init_components()
    
    def load_config(self) -> dict:
        """加载配置文件"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logging.error(f"配置文件不存在: {self.config_path}")
                sys.exit(1)
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logging.info(f"配置文件加载成功: {self.config_path}")
            return config
            
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """设置日志"""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('cs_detector.log', encoding='utf-8')
            ]
        )
        
        logging.info("日志系统初始化完成")
    
    def init_components(self):
        """初始化各个组件"""
        try:
            # 创建捕获器
            self.capture = create_capture_instance(self.config)
            logging.info("屏幕捕获器初始化完成")
            
            # 创建分析器
            self.analyzer = PlayerAnalyzer(self.config)
            logging.info("图像分析器初始化完成")
            
            # 创建输出管理器
            self.output_manager = OutputManager(self.config)
            logging.info("输出管理器初始化完成")
            
            # 启动HTTP API（如果启用）
            if self.config.get('output', {}).get('http_api', False):
                self.output_manager.start_http_server()
                logging.info("HTTP API服务器已启动")
            
        except Exception as e:
            logging.error(f"组件初始化失败: {e}")
            sys.exit(1)
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        logging.info(f"接收到信号 {signum}，正在停止...")
        self.stop()
    
    def start(self):
        """启动检测"""
        self.running = True
        self.start_time = time.time()
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logging.info("CS玩家检测器启动")
        
        # 获取检测频率配置
        fps = self.config.get('detection', {}).get('fps', 5)
        frame_interval = 1.0 / fps
        
        try:
            last_detection_time = 0
            
            while self.running:
                current_time = time.time()
                
                # 控制检测频率
                if current_time - last_detection_time < frame_interval:
                    time.sleep(0.01)  # 短暂休眠
                    continue
                
                # 执行一次检测
                success = self.detect_frame()
                
                if success:
                    last_detection_time = current_time
                    self.frame_count += 1
                    
                    # 每100帧输出一次性能统计
                    if self.frame_count % 100 == 0:
                        self.print_performance_stats()
                else:
                    # 检测失败时稍微等待
                    time.sleep(0.1)
        
        except Exception as e:
            logging.error(f"检测过程中发生错误: {e}")
        
        finally:
            self.cleanup()
    
    def detect_frame(self) -> bool:
        """
        检测单帧
        
        Returns:
            是否检测成功
        """
        try:
            # 捕获画面
            if hasattr(self.capture, 'get_screen_region'):
                frame = self.capture.get_screen_region()
            else:
                frame = self.capture.get_frame()
            
            if frame is None:
                logging.warning("未能捕获画面")
                return False
            
            # 提取玩家区域
            player_regions = self.capture.get_player_regions(frame)
            
            if not player_regions['team1'] and not player_regions['team2']:
                logging.warning("未检测到玩家区域")
                return False
            
            # 分析玩家信息
            results = self.analyzer.analyze_all_players(player_regions)
            
            # 输出结果
            self.output_manager.output_results(results)
            
            # 保存调试图像（如果启用）
            if self.debug_mode and self.save_debug_images:
                self.save_debug_frame(frame, player_regions, results)
            
            return True
            
        except Exception as e:
            logging.error(f"检测帧失败: {e}")
            return False
    
    def save_debug_frame(self, frame, player_regions, results):
        """保存调试图像"""
        try:
            # 保存原始帧
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            
            # 保存完整帧
            cv2.imwrite(f"debug_images/frame_{timestamp}.png", frame)
            
            # 保存每个玩家区域
            for team_name, team_players in player_regions.items():
                for i, player_data in enumerate(team_players):
                    region = player_data.get('region')
                    if region is not None and region.size > 0:
                        filename = f"debug_images/{team_name}_player{i+1}_{timestamp}.png"
                        cv2.imwrite(filename, region)
            
            # 创建并保存分析结果可视化
            debug_img = create_debug_image(player_regions, results)
            cv2.imwrite(f"debug_images/analysis_{timestamp}.png", debug_img)
            
        except Exception as e:
            logging.error(f"保存调试图像失败: {e}")
    
    def print_performance_stats(self):
        """打印性能统计"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            logging.info(f"性能统计 - 已处理帧数: {self.frame_count}, "
                        f"运行时间: {elapsed:.1f}s, "
                        f"平均FPS: {fps:.2f}")
    
    def stop(self):
        """停止检测"""
        self.running = False
        logging.info("正在停止检测器...")
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.capture:
                self.capture.cleanup()
            
            if self.output_manager:
                self.output_manager.cleanup()
            
            # 最终性能统计
            self.print_performance_stats()
            
            logging.info("CS玩家检测器已停止")
            
        except Exception as e:
            logging.error(f"清理资源失败: {e}")


def create_default_config():
    """创建默认配置文件"""
    default_config = {
        "capture": {
            "source": "screen",
            "monitor": 1,
            "region": {
                "x": 0,
                "y": 0,
                "width": 1920,
                "height": 1080
            }
        },
        "player_regions": {
            "team1": [
                {"x": 50, "y": 950, "width": 180, "height": 60, "name": "player1"},
                {"x": 250, "y": 950, "width": 180, "height": 60, "name": "player2"},
                {"x": 450, "y": 950, "width": 180, "height": 60, "name": "player3"},
                {"x": 650, "y": 950, "width": 180, "height": 60, "name": "player4"},
                {"x": 850, "y": 950, "width": 180, "height": 60, "name": "player5"}
            ],
            "team2": [
                {"x": 1080, "y": 950, "width": 180, "height": 60, "name": "player1"},
                {"x": 1280, "y": 950, "width": 180, "height": 60, "name": "player2"},
                {"x": 1480, "y": 950, "width": 180, "height": 60, "name": "player3"},
                {"x": 1680, "y": 950, "width": 180, "height": 60, "name": "player4"},
                {"x": 1880, "y": 950, "width": 180, "height": 60, "name": "player5"}
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
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print("默认配置文件 config.json 已创建")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CS直播画面玩家血量和存活状态检测工具')
    
    parser.add_argument('--source', type=str, help='输入源 (screen/视频文件路径)')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--save-debug', action='store_true', help='保存调试图像')
    parser.add_argument('--create-config', action='store_true', help='创建默认配置文件')
    parser.add_argument('--fps', type=int, help='检测帧率')
    parser.add_argument('--api', action='store_true', help='启用HTTP API')
    parser.add_argument('--port', type=int, help='HTTP API端口')
    
    args = parser.parse_args()
    
    # 创建默认配置
    if args.create_config:
        create_default_config()
        return
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"配置文件不存在: {args.config}")
        print("使用 --create-config 参数创建默认配置文件")
        return
    
    try:
        # 创建检测器
        detector = CSPlayerDetector(args.config)
        
        # 设置调试模式
        if args.debug:
            detector.debug_mode = True
            detector.save_debug_images = args.save_debug
            detector.setup_logging()  # 重新设置日志级别
        
        # 覆盖配置参数
        if args.source:
            detector.config['capture']['source'] = args.source
        
        if args.fps:
            detector.config['detection']['fps'] = args.fps
        
        if args.api:
            detector.config['output']['http_api'] = True
        
        if args.port:
            detector.config['output']['api_port'] = args.port
        
        # 重新初始化组件（如果有参数覆盖）
        if args.source or args.fps or args.api or args.port:
            detector.init_components()
        
        # 显示启动信息
        print("CS玩家检测器")
        print("=" * 30)
        print(f"输入源: {detector.config['capture']['source']}")
        print(f"检测频率: {detector.config['detection']['fps']} FPS")
        print(f"调试模式: {'开启' if detector.debug_mode else '关闭'}")
        print(f"HTTP API: {'开启' if detector.config['output']['http_api'] else '关闭'}")
        
        if detector.config['output']['http_api']:
            port = detector.config['output']['api_port']
            print(f"API地址: http://localhost:{port}/api/players")
        
        print("\n按 Ctrl+C 停止检测")
        print("-" * 30)
        
        # 启动检测
        detector.start()
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        logging.error(f"程序异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
