#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS直播画面捕获模块
负责从屏幕或视频流中捕获画面
"""

import cv2
import mss
import numpy as np
from PIL import Image
import time
import logging

class ScreenCapture:
    """屏幕捕获类"""
    
    def __init__(self, config):
        """
        初始化屏幕捕获器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.capture_config = config.get('capture', {})
        self.monitor = self.capture_config.get('monitor', 1)
        self.region = self.capture_config.get('region', {})
        
        # 初始化MSS屏幕捕获
        self.sct = mss.mss()
        
        logging.info(f"屏幕捕获器初始化完成，监视器: {self.monitor}")
    
    def get_screen_region(self):
        """
        获取指定屏幕区域的截图
        
        Returns:
            numpy.ndarray: BGR格式的图像数组
        """
        try:
            # 定义捕获区域
            if self.region:
                monitor = {
                    "top": self.region.get('y', 0),
                    "left": self.region.get('x', 0),
                    "width": self.region.get('width', 1920),
                    "height": self.region.get('height', 1080)
                }
            else:
                # 使用整个显示器
                monitor = self.sct.monitors[self.monitor]
            
            # 捕获屏幕
            screenshot = self.sct.grab(monitor)
            
            # 转换为numpy数组
            img = np.array(screenshot)
            
            # 转换颜色格式 BGRA -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
        except Exception as e:
            logging.error(f"屏幕捕获失败: {e}")
            return None
    
    def get_player_regions(self, frame):
        """
        从完整画面中提取玩家信息区域
        
        Args:
            frame: 完整的游戏画面
            
        Returns:
            dict: 包含两队玩家区域的字典
        """
        if frame is None:
            return {"team1": [], "team2": []}
        
        player_regions = {"team1": [], "team2": []}
        
        try:
            # 获取画面尺寸
            height, width = frame.shape[:2]
            
            # 提取每个队伍的玩家区域
            for team_name, team_regions in self.config['player_regions'].items():
                for i, region_config in enumerate(team_regions):
                    x = region_config.get('x', 0)
                    y = region_config.get('y', 0)
                    w = region_config.get('width', 100)
                    h = region_config.get('height', 50)
                    
                    # 确保坐标在有效范围内
                    x = max(0, min(x, width - w))
                    y = max(0, min(y, height - h))
                    
                    # 提取区域
                    region = frame[y:y+h, x:x+w]
                    
                    if region.size > 0:
                        player_regions[team_name].append({
                            'index': i,
                            'name': region_config.get('name', f'player{i+1}'),
                            'region': region,
                            'coords': (x, y, w, h)
                        })
            
        except Exception as e:
            logging.error(f"提取玩家区域失败: {e}")
        
        return player_regions
    
    def cleanup(self):
        """清理资源"""
        try:
            self.sct.close()
            logging.info("屏幕捕获器资源清理完成")
        except Exception as e:
            logging.error(f"清理屏幕捕获器资源失败: {e}")


class VideoCapture:
    """视频文件或流捕获类（用于测试或离线分析）"""
    
    def __init__(self, source_path):
        """
        初始化视频捕获器
        
        Args:
            source_path: 视频文件路径或流URL
        """
        self.source_path = source_path
        self.cap = cv2.VideoCapture(source_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {source_path}")
        
        logging.info(f"视频捕获器初始化完成，源: {source_path}")
    
    def get_frame(self):
        """
        获取下一帧
        
        Returns:
            numpy.ndarray: BGR格式的图像数组，如果失败返回None
        """
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
            logging.info("视频捕获器资源清理完成")


def create_capture_instance(config):
    """
    创建捕获实例的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        捕获实例
    """
    source = config.get('capture', {}).get('source', 'screen')
    
    if source == 'screen':
        return ScreenCapture(config)
    else:
        # 假设是视频文件或流
        return VideoCapture(source)


if __name__ == "__main__":
    # 测试代码
    import json
    
    # 加载配置
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建捕获实例
    capture = create_capture_instance(config)
    
    if isinstance(capture, ScreenCapture):
        # 测试屏幕捕获
        frame = capture.get_screen_region()
        if frame is not None:
            print(f"捕获成功，画面尺寸: {frame.shape}")
            
            # 提取玩家区域
            player_regions = capture.get_player_regions(frame)
            print(f"提取到的玩家区域数量 - Team1: {len(player_regions['team1'])}, Team2: {len(player_regions['team2'])}")
        
        capture.cleanup()