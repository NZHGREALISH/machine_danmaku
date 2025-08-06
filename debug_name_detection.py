#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
玩家名称识别调试脚本
专门分析名称识别的每个步骤
"""

import cv2
import numpy as np
import pytesseract
import json
import sys
from pathlib import Path
import re

class NameDetectionDebugger:
    """名称检测调试器"""
    
    def __init__(self, config_path="config.json"):
        """初始化调试器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.ocr_config = self.config.get('ocr', {})
        self.name_offset = self.ocr_config.get('name_region_offset', {'x': 80, 'y': 45, 'width': 100, 'height': 25})
    
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
    
    def extract_name_region(self, player_region):
        """提取名称区域"""
        if player_region is None or player_region.size == 0:
            return None
        
        h, w = player_region.shape[:2]
        
        # 使用配置的偏移量
        x = min(self.name_offset['x'], w - self.name_offset['width'])
        y = min(self.name_offset['y'], h - self.name_offset['height'])
        x = max(0, x)
        y = max(0, y)
        
        name_width = min(self.name_offset['width'], w - x)
        name_height = min(self.name_offset['height'], h - y)
        
        if name_width <= 0 or name_height <= 0:
            return None
        
        # 提取名称区域
        name_region = player_region[y:y+name_height, x:x+name_width]
        return name_region
    
    def preprocess_name_image_step_by_step(self, image):
        """逐步预处理名称图像"""
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
        
        # 对比度增强
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=20)
        steps['enhanced'] = enhanced
        
        # 反色处理
        inverted = cv2.bitwise_not(gray)
        steps['inverted'] = inverted
        
        # 二值化 - OTSU
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        steps['binary_otsu'] = binary_otsu
        
        # 二值化 - 固定阈值
        _, binary_fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        steps['binary_fixed'] = binary_fixed
        
        # 自适应二值化
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        steps['adaptive'] = adaptive
        
        # 形态学操作 - 去噪
        kernel = np.ones((2, 2), np.uint8)
        morph_open = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel)
        steps['morph_open'] = morph_open
        
        # 形态学操作 - 闭运算
        morph_close = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)
        steps['morph_close'] = morph_close
        
        return steps
    
    def test_name_ocr_methods(self, image):
        """测试不同的名称OCR方法"""
        results = {}
        
        if image is None or image.size == 0:
            return results
        
        # OCR配置列表
        ocr_configs = [
            ("default", "--psm 8"),
            ("single_word", "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"),
            ("line_text", "--psm 7"),
            ("line_text_whitelist", "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"),
            ("word_text", "--psm 6"),
            ("sparse_text", "--psm 11"),
            ("raw_line", "--psm 13"),
        ]
        
        for config_name, config_str in ocr_configs:
            try:
                text = pytesseract.image_to_string(image, config=config_str).strip()
                
                # 清理文本
                cleaned = re.sub(r'[^a-zA-Z0-9_\-]', '', text)
                
                results[config_name] = {
                    'raw_text': text,
                    'cleaned': cleaned,
                    'length': len(cleaned),
                    'valid': len(cleaned) >= 2 and len(cleaned) <= 20
                }
            except Exception as e:
                results[config_name] = {'error': str(e)}
        
        return results
    
    def debug_player_name(self, player_data, team_name, save_debug=True):
        """调试单个玩家的名称识别"""
        player_name = player_data.get('name', 'unknown')
        player_region = player_data.get('region')
        coords = player_data.get('coords', (0, 0, 0, 0))
        
        print(f"\n🔍 调试 {team_name} - {player_name}")
        print(f"   区域坐标: {coords}")
        print(f"   区域尺寸: {player_region.shape if player_region is not None else 'None'}")
        
        if player_region is None or player_region.size == 0:
            print("   ❌ 玩家区域为空")
            return
        
        # 提取名称区域
        name_region = self.extract_name_region(player_region)
        
        if name_region is None or name_region.size == 0:
            print("   ❌ 名称区域提取失败")
            print(f"   名称偏移配置: {self.name_offset}")
            return
        
        print(f"   名称区域尺寸: {name_region.shape}")
        print(f"   名称偏移: x={self.name_offset['x']}, y={self.name_offset['y']}")
        
        # 逐步预处理
        preprocessing_steps = self.preprocess_name_image_step_by_step(name_region)
        
        # 测试不同OCR方法
        print("   OCR测试结果:")
        best_results = []
        
        for step_name, processed_image in preprocessing_steps.items():
            if step_name in ['grayscale', 'enhanced', 'inverted', 'binary_otsu', 'adaptive']:
                ocr_results = self.test_name_ocr_methods(processed_image)
                
                print(f"     {step_name}:")
                for method_name, result in ocr_results.items():
                    if 'error' not in result:
                        raw = result.get('raw_text', '').replace('\n', '\\n')
                        cleaned = result.get('cleaned', '')
                        valid = result.get('valid', False)
                        
                        print(f"       {method_name}: raw='{raw}' cleaned='{cleaned}' valid={valid}")
                        
                        if valid and cleaned:
                            confidence = self._calculate_simple_confidence(raw, cleaned)
                            best_results.append({
                                'method': f"{step_name}_{method_name}",
                                'name': cleaned,
                                'confidence': confidence,
                                'raw': raw
                            })
                    else:
                        print(f"       {method_name}: ERROR")
        
        # 显示最佳结果
        if best_results:
            best_results.sort(key=lambda x: x['confidence'], reverse=True)
            print(f"   🎯 最佳结果:")
            for i, result in enumerate(best_results[:3]):  # 显示前3个
                print(f"     {i+1}. {result['method']}: '{result['name']}' (confidence: {result['confidence']:.1f})")
        else:
            print("   ❌ 所有OCR方法都失败了")
        
        # 保存调试图像
        if save_debug:
            debug_dir = Path("name_debug")
            debug_dir.mkdir(exist_ok=True)
            
            # 保存玩家区域
            cv2.imwrite(f"name_debug/{team_name}_{player_name}_player_region.png", player_region)
            
            # 保存名称区域
            cv2.imwrite(f"name_debug/{team_name}_{player_name}_name_region.png", name_region)
            
            # 保存预处理步骤
            for step_name, processed_image in preprocessing_steps.items():
                cv2.imwrite(f"name_debug/{team_name}_{player_name}_{step_name}.png", processed_image)
            
            # 创建带名称区域标注的图像
            annotated = player_region.copy()
            x, y, w, h = self.name_offset['x'], self.name_offset['y'], self.name_offset['width'], self.name_offset['height']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "NAME", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(f"name_debug/{team_name}_{player_name}_annotated.png", annotated)
    
    def _calculate_simple_confidence(self, raw_text, cleaned_text):
        """简单的置信度计算"""
        confidence = 0.0
        
        if cleaned_text:
            confidence += 30.0
        
        # 长度合理性
        if 3 <= len(cleaned_text) <= 15:
            confidence += 25.0
        elif 2 <= len(cleaned_text) <= 20:
            confidence += 15.0
        
        # 清洁度
        if raw_text:
            clean_ratio = len(cleaned_text) / max(len(raw_text), 1)
            confidence += clean_ratio * 20.0
        
        # 字符合理性
        if re.search(r'[a-zA-Z]', cleaned_text):
            confidence += 15.0
        
        # 避免纯数字
        if cleaned_text.isdigit():
            confidence -= 10.0
        
        return confidence
    
    def debug_image(self, image_path):
        """调试整张图片的名称识别"""
        print(f"🎯 开始调试名称识别: {image_path}")
        
        # 加载图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法加载图片: {image_path}")
            return
        
        print(f"✅ 图片加载成功，尺寸: {image.shape}")
        print(f"🔧 当前名称偏移配置: {self.name_offset}")
        
        # 提取玩家区域
        player_regions = self.extract_player_regions_from_image(image)
        
        # 调试每个玩家
        for team_name, team_players in player_regions.items():
            print(f"\n{'='*50}")
            print(f"🏆 {team_name.upper()}")
            print(f"{'='*50}")
            
            for player_data in team_players:
                self.debug_player_name(player_data, team_name)
        
        print(f"\n💾 详细调试图像已保存到 name_debug/ 目录")
        print(f"📝 建议:")
        print(f"   1. 查看 name_debug/ 中的图像，确认名称区域是否正确")
        print(f"   2. 如果名称区域不对，调整 config.json 中的 name_region_offset")
        print(f"   3. 名称区域应该只包含玩家名称文字，不包含其他元素")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python debug_name_detection.py <图片路径>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"❌ 图片文件不存在: {image_path}")
        sys.exit(1)
    
    debugger = NameDetectionDebugger()
    debugger.debug_image(image_path)


if __name__ == "__main__":
    main()