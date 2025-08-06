#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
血量检测调试脚本
详细分析血量检测的每个步骤
"""

import cv2
import numpy as np
import pytesseract
import json
import sys
from pathlib import Path
import re

class HPDetectionDebugger:
    """血量检测调试器"""
    
    def __init__(self, config_path="config.json"):
        """初始化调试器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.ocr_config = self.config.get('ocr', {})
        self.tesseract_config = self.ocr_config.get('tesseract_config', '--psm 8 -c tessedit_char_whitelist=0123456789')
        self.hp_offset = self.ocr_config.get('hp_region_offset', {'x': 87, 'y': 1, 'width': 55, 'height': 46})
    
    def extract_player_regions_from_image(self, image):
        """从图片中提取玩家区域"""
        player_regions = {"team1": [], "team2": []}
        
        height, width = image.shape[:2]
        
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
                region = image[y:y+h, x:x+w]
                
                if region.size > 0:
                    player_regions[team_name].append({
                        'index': i,
                        'name': region_config.get('name', f'player{i+1}'),
                        'region': region,
                        'coords': (x, y, w, h)
                    })
        
        return player_regions
    
    def extract_hp_region(self, player_region):
        """提取血量数字区域"""
        if player_region is None or player_region.size == 0:
            return None
        
        h, w = player_region.shape[:2]
        
        # 使用配置的偏移量
        x = min(self.hp_offset['x'], w - self.hp_offset['width'])
        y = min(self.hp_offset['y'], h - self.hp_offset['height'])
        x = max(0, x)
        y = max(0, y)
        
        hp_width = min(self.hp_offset['width'], w - x)
        hp_height = min(self.hp_offset['height'], h - y)
        
        if hp_width <= 0 or hp_height <= 0:
            return None
        
        # 提取血量区域
        hp_region = player_region[y:y+hp_height, x:x+hp_width]
        return hp_region
    
    def preprocess_image_step_by_step(self, image):
        """逐步预处理图像，返回每个步骤的结果"""
        steps = {}
        
        if image is None or image.size == 0:
            return steps
        
        # 原始图像
        steps['original'] = image.copy()
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        steps['grayscale'] = gray
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        steps['blurred'] = blurred
        
        # 对比度增强
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=10)
        steps['enhanced'] = enhanced
        
        # 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        steps['binary'] = binary
        
        # 尝试不同的二值化方法
        _, binary_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        steps['binary_simple'] = binary_simple
        
        # 自适应二值化
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        steps['adaptive'] = adaptive
        
        return steps
    
    def test_ocr_methods(self, image):
        """测试不同的OCR方法"""
        results = {}
        
        if image is None or image.size == 0:
            return results
        
        # 方法1: 默认配置
        try:
            text = pytesseract.image_to_string(image, config=self.tesseract_config)
            numbers = re.findall(r'\d+', text)
            results['default'] = {
                'text': text.strip(),
                'numbers': numbers,
                'hp': int(numbers[0]) if numbers else 0
            }
        except Exception as e:
            results['default'] = {'error': str(e)}
        
        # 方法2: 不同的PSM模式
        psm_modes = [6, 7, 8, 10, 13]
        for psm in psm_modes:
            try:
                config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(image, config=config)
                numbers = re.findall(r'\d+', text)
                results[f'psm_{psm}'] = {
                    'text': text.strip(),
                    'numbers': numbers,
                    'hp': int(numbers[0]) if numbers else 0
                }
            except Exception as e:
                results[f'psm_{psm}'] = {'error': str(e)}
        
        # 方法3: 数字识别
        try:
            config = "--psm 8 outputbase digits"
            text = pytesseract.image_to_string(image, config=config)
            numbers = re.findall(r'\d+', text)
            results['digits_only'] = {
                'text': text.strip(),
                'numbers': numbers,
                'hp': int(numbers[0]) if numbers else 0
            }
        except Exception as e:
            results['digits_only'] = {'error': str(e)}
        
        return results
    
    def debug_player(self, player_data, team_name, save_debug=True):
        """调试单个玩家的血量检测"""
        player_name = player_data.get('name', 'unknown')
        player_region = player_data.get('region')
        coords = player_data.get('coords', (0, 0, 0, 0))
        
        print(f"\n🔍 调试 {team_name} - {player_name}")
        print(f"   区域坐标: {coords}")
        print(f"   区域尺寸: {player_region.shape if player_region is not None else 'None'}")
        
        if player_region is None or player_region.size == 0:
            print("   ❌ 玩家区域为空")
            return
        
        # 提取血量区域
        hp_region = self.extract_hp_region(player_region)
        
        if hp_region is None or hp_region.size == 0:
            print("   ❌ 血量区域提取失败")
            print(f"   血量偏移配置: {self.hp_offset}")
            return
        
        print(f"   血量区域尺寸: {hp_region.shape}")
        print(f"   血量偏移: x={self.hp_offset['x']}, y={self.hp_offset['y']}")
        
        # 逐步预处理
        preprocessing_steps = self.preprocess_image_step_by_step(hp_region)
        
        # 测试不同OCR方法
        ocr_results = {}
        for step_name, processed_image in preprocessing_steps.items():
            if step_name in ['grayscale', 'binary', 'binary_simple', 'adaptive', 'enhanced']:
                ocr_results[step_name] = self.test_ocr_methods(processed_image)
        
        # 显示OCR结果
        print("   OCR测试结果:")
        best_result = None
        best_confidence = 0
        
        for preprocess_method, ocr_methods in ocr_results.items():
            print(f"     {preprocess_method}:")
            for method_name, result in ocr_methods.items():
                if 'error' not in result:
                    text = result.get('text', '').replace('\n', '\\n')
                    numbers = result.get('numbers', [])
                    hp = result.get('hp', 0)
                    print(f"       {method_name}: text='{text}' numbers={numbers} hp={hp}")
                    
                    # 简单的置信度评估
                    confidence = len(numbers) * 10 + (10 if hp > 0 and hp <= 100 else 0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {'method': f"{preprocess_method}_{method_name}", 'hp': hp}
                else:
                    print(f"       {method_name}: ERROR - {result['error']}")
        
        if best_result:
            print(f"   🎯 最佳结果: {best_result['method']} -> HP={best_result['hp']}")
        else:
            print("   ❌ 所有OCR方法都失败了")
        
        # 保存调试图像
        if save_debug:
            debug_dir = Path("hp_debug")
            debug_dir.mkdir(exist_ok=True)
            
            # 保存玩家区域
            cv2.imwrite(f"hp_debug/{team_name}_{player_name}_player_region.png", player_region)
            
            # 保存血量区域
            cv2.imwrite(f"hp_debug/{team_name}_{player_name}_hp_region.png", hp_region)
            
            # 保存预处理步骤
            for step_name, processed_image in preprocessing_steps.items():
                cv2.imwrite(f"hp_debug/{team_name}_{player_name}_{step_name}.png", processed_image)
            
            # 创建带血量区域标注的图像
            annotated = player_region.copy()
            x, y, w, h = self.hp_offset['x'], self.hp_offset['y'], self.hp_offset['width'], self.hp_offset['height']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(f"hp_debug/{team_name}_{player_name}_annotated.png", annotated)
    
    def debug_image(self, image_path):
        """调试整张图片的血量检测"""
        print(f"🎯 开始调试血量检测: {image_path}")
        
        # 加载图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法加载图片: {image_path}")
            return
        
        print(f"✅ 图片加载成功，尺寸: {image.shape}")
        print(f"🔧 当前血量偏移配置: {self.hp_offset}")
        
        # 提取玩家区域
        player_regions = self.extract_player_regions_from_image(image)
        
        # 调试每个玩家
        for team_name, team_players in player_regions.items():
            print(f"\n{'='*50}")
            print(f"🏆 {team_name.upper()}")
            print(f"{'='*50}")
            
            for player_data in team_players:
                self.debug_player(player_data, team_name)
        
        print(f"\n💾 详细调试图像已保存到 hp_debug/ 目录")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python debug_hp_detection.py <图片路径>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"❌ 图片文件不存在: {image_path}")
        sys.exit(1)
    
    debugger = HPDetectionDebugger()
    debugger.debug_image(image_path)


if __name__ == "__main__":
    main()