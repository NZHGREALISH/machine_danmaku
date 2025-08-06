#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS直播画面分析模块
负责图像预处理、OCR识别和存活状态判断
"""

import cv2
import numpy as np
import pytesseract
import re
import logging
from typing import Dict, List, Tuple, Optional

class PlayerAnalyzer:
    """玩家信息分析器"""
    
    def __init__(self, config):
        """
        初始化分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.ocr_config = config.get('ocr', {})
        self.detection_config = config.get('detection', {})
        
        # OCR配置
        self.tesseract_config = self.ocr_config.get('tesseract_config', '--psm 8 -c tessedit_char_whitelist=0123456789')
        self.name_tesseract_config = self.ocr_config.get('name_tesseract_config', '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-')
        self.hp_offset = self.ocr_config.get('hp_region_offset', {'x': 87, 'y': 1, 'width': 55, 'height': 46})
        self.name_offset = self.ocr_config.get('name_region_offset', {'x': 3, 'y': 50, 'width': 135, 'height': 20})
        
        # 检测阈值
        self.alive_threshold = self.detection_config.get('alive_threshold', 0.3)
        self.grayscale_threshold = self.detection_config.get('grayscale_threshold', 100)
        
        logging.info("玩家分析器初始化完成")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        if image is None or image.size == 0:
            return None
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 对比度增强
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=10)
        
        # 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_hp_region(self, player_region: np.ndarray) -> Optional[np.ndarray]:
        """
        提取血量数字区域
        
        Args:
            player_region: 玩家信息区域
            
        Returns:
            血量数字区域图像
        """
        if player_region is None or player_region.size == 0:
            return None
        
        try:
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
            
        except Exception as e:
            logging.warning(f"提取血量区域失败: {e}")
            return None
    
    def recognize_hp_value(self, hp_region: np.ndarray) -> int:
        """
        识别血量数值
        
        Args:
            hp_region: 血量区域图像
            
        Returns:
            血量数值 (0-100)
        """
        if hp_region is None or hp_region.size == 0:
            return 0
        
        try:
            # 使用多种方法尝试识别血量
            hp_candidates = []
            
            # 方法1: 灰度图直接识别
            if len(hp_region.shape) == 3:
                gray = cv2.cvtColor(hp_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = hp_region.copy()
            
            hp_candidates.extend(self._try_ocr_methods(gray))
            
            # 方法2: 对比度增强后识别
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
            hp_candidates.extend(self._try_ocr_methods(enhanced))
            
            # 方法3: 二值化后识别
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            hp_candidates.extend(self._try_ocr_methods(binary))
            
            # 方法4: 自适应二值化
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            hp_candidates.extend(self._try_ocr_methods(adaptive))
            
            # 选择最可信的结果
            final_hp = self._select_best_hp_result(hp_candidates, hp_region)
            
            return final_hp
                
        except Exception as e:
            logging.warning(f"血量识别失败: {e}")
            return 0
    
    def _try_ocr_methods(self, image: np.ndarray) -> list:
        """
        尝试多种OCR方法
        
        Args:
            image: 预处理后的图像
            
        Returns:
            候选结果列表
        """
        candidates = []
        
        # 不同的PSM模式
        psm_modes = [6, 7, 8, 10, 13]
        
        for psm in psm_modes:
            try:
                config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(image, config=config).strip()
                
                # 提取数字
                numbers = re.findall(r'\d+', text)
                if numbers:
                    hp = int(numbers[0])
                    if 0 <= hp <= 100:  # 只接受合理范围的数值
                        candidates.append({
                            'hp': hp,
                            'confidence': self._calculate_confidence(text, numbers),
                            'method': f'psm_{psm}',
                            'text': text
                        })
            except Exception:
                continue
        
        return candidates
    
    def _calculate_confidence(self, text: str, numbers: list) -> float:
        """
        计算OCR结果的置信度
        
        Args:
            text: OCR识别的文本
            numbers: 提取的数字列表
            
        Returns:
            置信度分数
        """
        confidence = 0.0
        
        # 基础分数：有数字就给分
        if numbers:
            confidence += 30.0
        
        # 数字长度合理性
        if numbers and len(numbers[0]) <= 3:  # 血量最多3位数
            confidence += 20.0
        
        # 文本清洁度（越少噪音字符越好）
        clean_ratio = len(''.join(numbers)) / max(len(text), 1)
        confidence += clean_ratio * 30.0
        
        # 数值合理性
        if numbers:
            hp = int(numbers[0])
            if hp == 0:
                confidence += 10.0  # 0是常见值
            elif 1 <= hp <= 100:
                confidence += 20.0  # 正常血量范围
        
        return confidence
    
    def _select_best_hp_result(self, candidates: list, hp_region: np.ndarray) -> int:
        """
        选择最佳的血量识别结果
        
        Args:
            candidates: 候选结果列表
            hp_region: 血量区域图像
            
        Returns:
            最终血量值
        """
        if not candidates:
            # 如果所有OCR都失败，检查是否为空血量
            return self._check_empty_hp(hp_region)
        
        # 按置信度排序
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 特殊处理：如果多个结果都指向0，优先选择0
        zero_candidates = [c for c in candidates if c['hp'] == 0]
        if len(zero_candidates) >= 2:  # 至少两个方法都认为是0
            return 0
        
        # 特殊处理：检查图像是否看起来像空血量
        if self._looks_like_empty_hp(hp_region):
            # 如果图像看起来是空的，优先选择0或最小值
            low_hp_candidates = [c for c in candidates if c['hp'] <= 10]
            if low_hp_candidates:
                return 0
        
        # 返回置信度最高的结果
        best_candidate = candidates[0]
        
        # 额外验证：如果最佳结果是小数值但图像看起来不像低血量，可能是误读
        if best_candidate['hp'] <= 10 and not self._looks_like_low_hp(hp_region):
            # 检查是否有其他合理的候选
            other_candidates = [c for c in candidates[1:] if c['hp'] > 10 and c['confidence'] > 30]
            if other_candidates:
                return other_candidates[0]['hp']
        
        return best_candidate['hp']
    
    def _check_empty_hp(self, hp_region: np.ndarray) -> int:
        """
        检查是否为空血量区域
        
        Args:
            hp_region: 血量区域图像
            
        Returns:
            如果看起来是空的返回0，否则返回-1表示不确定
        """
        try:
            if hp_region is None or hp_region.size == 0:
                return 0
            
            # 转换为灰度
            if len(hp_region.shape) == 3:
                gray = cv2.cvtColor(hp_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = hp_region.copy()
            
            # 计算像素强度分布
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # 如果区域很均匀且较暗，可能是空血量
            if std_intensity < 10 and mean_intensity < 50:
                return 0
            
            # 检查边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 如果几乎没有边缘，可能是空区域
            if edge_density < 0.05:
                return 0
            
            return -1  # 不确定
            
        except Exception:
            return 0
    
    def _looks_like_empty_hp(self, hp_region: np.ndarray) -> bool:
        """
        检查图像是否看起来像空血量
        
        Args:
            hp_region: 血量区域图像
            
        Returns:
            是否看起来是空的
        """
        try:
            if hp_region is None or hp_region.size == 0:
                return True
            
            # 转换为灰度
            if len(hp_region.shape) == 3:
                gray = cv2.cvtColor(hp_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = hp_region.copy()
            
            # 检查亮像素比例
            bright_pixels = np.sum(gray > 200)
            total_pixels = gray.size
            bright_ratio = bright_pixels / total_pixels
            
            # 如果亮像素很少，可能是空的
            return bright_ratio < 0.1
            
        except Exception:
            return False
    
    def _looks_like_low_hp(self, hp_region: np.ndarray) -> bool:
        """
        检查图像是否看起来像低血量
        
        Args:
            hp_region: 血量区域图像
            
        Returns:
            是否看起来是低血量
        """
        try:
            if hp_region is None or hp_region.size == 0:
                return True
            
            # 转换为灰度
            if len(hp_region.shape) == 3:
                gray = cv2.cvtColor(hp_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = hp_region.copy()
            
            # 检查数字区域的特征
            # 低血量通常是1-2位数，占用空间较小
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 计算轮廓的总面积
                total_area = sum(cv2.contourArea(c) for c in contours)
                image_area = gray.shape[0] * gray.shape[1]
                area_ratio = total_area / image_area
                
                # 如果轮廓面积相对较小，可能是低血量
                return area_ratio < 0.2
            
            return False
            
        except Exception:
            return False
    
    def _fallback_hp_detection(self, processed_image: np.ndarray) -> int:
        """
        OCR失败时的备用血量检测方法
        
        Args:
            processed_image: 预处理后的图像
            
        Returns:
            估计的血量值
        """
        try:
            # 基于像素密度的简单估计
            white_pixels = np.sum(processed_image == 255)
            total_pixels = processed_image.size
            
            if total_pixels == 0:
                return 0
            
            density = white_pixels / total_pixels
            
            # 简单的密度到血量映射
            if density > 0.3:
                return int(density * 100)
            else:
                return 0
                
        except Exception:
            return 0
    
    def is_player_alive(self, player_region: np.ndarray, hp_value: int) -> bool:
        """
        判断玩家是否存活
        
        Args:
            player_region: 玩家信息区域
            hp_value: 血量值
            
        Returns:
            是否存活
        """
        # 最重要的判断：血量为0必定死亡
        if hp_value == 0:
            return False
        
        # 如果血量>0，通常是存活的
        if hp_value > 0:
            return True
        
        # 这种情况理论上不应该出现（hp_value为负数），
        # 但为了安全起见，使用图像灰度作为备用判断
        try:
            if player_region is None or player_region.size == 0:
                return False
            
            # 计算平均灰度值
            if len(player_region.shape) == 3:
                gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = player_region
            
            mean_gray = np.mean(gray)
            
            # 如果平均灰度值低于阈值，认为玩家已死亡
            return mean_gray > self.grayscale_threshold
            
        except Exception as e:
            logging.warning(f"存活状态判断失败: {e}")
            return False  # 出错时默认为死亡
    
    def extract_name_region(self, player_region: np.ndarray) -> Optional[np.ndarray]:
        """
        提取玩家名称区域
        
        Args:
            player_region: 玩家信息区域
            
        Returns:
            玩家名称区域图像
        """
        if player_region is None or player_region.size == 0:
            return None
        
        try:
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
            
        except Exception as e:
            logging.warning(f"提取名称区域失败: {e}")
            return None

    def extract_player_name(self, player_region: np.ndarray) -> str:
        """
        提取玩家名称 - 优化版本，基于测试数据的最佳策略
        
        Args:
            player_region: 玩家信息区域
            
        Returns:
            玩家名称
        """
        try:
            if player_region is None or player_region.size == 0:
                return "Unknown"
            
            # 提取名称区域
            name_region = self.extract_name_region(player_region)
            
            if name_region is None or name_region.size == 0:
                return "Unknown"
            
            # 转换为灰度图
            if len(name_region.shape) == 3:
                gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = name_region.copy()
            
            # 使用基于测试数据的最佳策略
            best_results = []
            
            # 策略1: 反色 + 单词模式 (最高优先级 - 80%成功率)
            inverted = cv2.bitwise_not(gray)
            result1 = self._try_optimized_ocr(inverted, "single_word")
            if result1:
                best_results.append(('inverted_single_word', result1, 100))
            
            # 策略2: 灰度 + 单词模式 (中优先级 - 20%成功率)
            result2 = self._try_optimized_ocr(gray, "single_word")
            if result2:
                best_results.append(('grayscale_single_word', result2, 80))
            
            # 策略3: 反色 + 行文本模式 (备用策略)
            result3 = self._try_optimized_ocr(inverted, "line_text_whitelist")
            if result3:
                best_results.append(('inverted_line_text', result3, 60))
            
            # 策略4: 二值化 + 单词模式 (最后备用)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result4 = self._try_optimized_ocr(binary, "single_word")
            if result4:
                best_results.append(('binary_single_word', result4, 40))
            
            # 选择最佳结果
            if best_results:
                # 按优先级排序
                best_results.sort(key=lambda x: x[2], reverse=True)
                best_name = best_results[0][1]
                logging.debug(f"名称识别成功: {best_name} (方法: {best_results[0][0]})")
                return best_name
            
            return "Unknown"
            
        except Exception as e:
            logging.warning(f"玩家名称提取失败: {e}")
            return "Unknown"
    
    def _try_optimized_ocr(self, image: np.ndarray, mode: str) -> str:
        """
        使用优化的OCR方法
        
        Args:
            image: 预处理后的图像
            mode: OCR模式 ('single_word', 'line_text_whitelist')
            
        Returns:
            识别出的名称，失败返回空字符串
        """
        try:
            if mode == "single_word":
                config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            elif mode == "line_text_whitelist":
                config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            else:
                return ""
            
            text = pytesseract.image_to_string(image, config=config).strip()
            
            # 清理文本
            cleaned = re.sub(r'[^a-zA-Z0-9_\-]', '', text)
            
            # 验证结果质量
            if self._is_valid_player_name(cleaned):
                return cleaned
            
            return ""
            
        except Exception:
            return ""
    
    def _is_valid_player_name(self, name: str) -> bool:
        """
        验证玩家名称是否有效
        
        Args:
            name: 候选名称
            
        Returns:
            是否为有效的玩家名称
        """
        if not name or len(name) < 2:
            return False
        
        # 长度限制
        if len(name) > 20:
            return False
        
        # 必须包含字母
        if not re.search(r'[a-zA-Z]', name):
            return False
        
        # 避免纯重复字符
        if len(set(name.lower())) < 2:
            return False
        
        # 避免明显的垃圾字符组合
        garbage_patterns = [
            r'^[aeiou]{3,}$',  # 纯元音
            r'^[xyz]{2,}$',    # 重复xyz
            r'^[0-9]{4,}$',    # 纯数字太长
        ]
        
        for pattern in garbage_patterns:
            if re.match(pattern, name.lower()):
                return False
        
        return True
    
    def analyze_player(self, player_data: Dict) -> Dict:
        """
        分析单个玩家的信息
        
        Args:
            player_data: 包含玩家区域信息的字典
            
        Returns:
            玩家分析结果
        """
        result = {
            'index': player_data.get('index', 0),
            'name': player_data.get('name', 'Unknown'),
            'hp': 0,
            'alive': False,
            'coords': player_data.get('coords', (0, 0, 0, 0))
        }
        
        try:
            player_region = player_data.get('region')
            
            if player_region is None or player_region.size == 0:
                return result
            
            # 提取血量区域
            hp_region = self.extract_hp_region(player_region)
            
            # 识别血量
            hp_value = self.recognize_hp_value(hp_region)
            result['hp'] = hp_value
            
            # 判断存活状态
            alive = self.is_player_alive(player_region, hp_value)
            result['alive'] = alive
            
            # 提取玩家名称
            try:
                extracted_name = self.extract_player_name(player_region)
                if extracted_name and extracted_name != "Unknown":
                    result['name'] = extracted_name
                    logging.debug(f"提取到玩家名称: {extracted_name}")
                else:
                    # 保留默认名称
                    logging.debug(f"未能提取玩家名称，使用默认: {result['name']}")
            except Exception as e:
                logging.warning(f"玩家名称提取异常: {e}")
            
        except Exception as e:
            logging.error(f"分析玩家 {result['name']} 失败: {e}")
        
        return result
    
    def analyze_all_players(self, player_regions: Dict) -> Dict:
        """
        分析所有玩家信息
        
        Args:
            player_regions: 所有玩家区域数据
            
        Returns:
            完整的分析结果
        """
        results = {
            'team1': [],
            'team2': [],
            'timestamp': None
        }
        
        try:
            import time
            results['timestamp'] = int(time.time() * 1000)  # 毫秒时间戳
            
            # 分析每个队伍的玩家
            for team_name in ['team1', 'team2']:
                team_players = player_regions.get(team_name, [])
                
                for player_data in team_players:
                    player_result = self.analyze_player(player_data)
                    results[team_name].append(player_result)
                    
                    logging.debug(f"{team_name} {player_result['name']}: HP={player_result['hp']}, Alive={player_result['alive']}")
            
        except Exception as e:
            logging.error(f"分析所有玩家失败: {e}")
        
        return results


def create_debug_image(player_regions: Dict, analysis_results: Dict) -> np.ndarray:
    """
    创建调试可视化图像
    
    Args:
        player_regions: 玩家区域数据
        analysis_results: 分析结果
        
    Returns:
        调试图像
    """
    try:
        # 创建空白画布
        debug_img = np.ones((400, 800, 3), dtype=np.uint8) * 255
        
        y_offset = 20
        
        # 显示分析结果
        for team_name in ['team1', 'team2']:
            cv2.putText(debug_img, f"{team_name.upper()}:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += 30
            
            team_results = analysis_results.get(team_name, [])
            for player in team_results:
                status = "ALIVE" if player['alive'] else "DEAD"
                color = (0, 255, 0) if player['alive'] else (0, 0, 255)
                
                text = f"  {player['name']}: HP={player['hp']} ({status})"
                cv2.putText(debug_img, text, (40, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
            
            y_offset += 20
        
        return debug_img
        
    except Exception as e:
        logging.error(f"创建调试图像失败: {e}")
        return np.ones((400, 800, 3), dtype=np.uint8) * 255


if __name__ == "__main__":
    # 测试代码
    import json
    from capture import create_capture_instance
    
    # 加载配置
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建分析器
    analyzer = PlayerAnalyzer(config)
    
    # 创建捕获器
    capture = create_capture_instance(config)
    
    try:
        # 获取一帧进行测试
        if hasattr(capture, 'get_screen_region'):
            frame = capture.get_screen_region()
        else:
            frame = capture.get_frame()
        
        if frame is not None:
            # 提取玩家区域
            player_regions = capture.get_player_regions(frame)
            
            # 分析玩家信息
            results = analyzer.analyze_all_players(player_regions)
            
            print("分析结果:")
            print(json.dumps(results, indent=2, ensure_ascii=False))
            
            # 创建调试图像
            debug_img = create_debug_image(player_regions, results)
            cv2.imshow("Debug", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    finally:
        capture.cleanup()