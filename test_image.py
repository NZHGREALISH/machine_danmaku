#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS直播画面检测工具 - 图片测试脚本
用于测试单张图片的检测效果
"""

import cv2
import json
import sys
import argparse
from pathlib import Path
import logging

# 导入自定义模块
from analyzer import PlayerAnalyzer, create_debug_image
from output import OutputManager, ResultFormatter

class ImageTester:
    """图片测试器"""
    
    def __init__(self, config_path="config.json"):
        """初始化测试器"""
        self.config = self.load_config(config_path)
        self.analyzer = PlayerAnalyzer(self.config)
        self.setup_logging()
    
    def load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def extract_player_regions_from_image(self, image):
        """从图片中提取玩家区域"""
        if image is None:
            return {"team1": [], "team2": []}
        
        player_regions = {"team1": [], "team2": []}
        
        try:
            height, width = image.shape[:2]
            print(f"📸 图片尺寸: {width}x{height}")
            
            # 提取每个队伍的玩家区域
            for team_name, team_regions in self.config['player_regions'].items():
                print(f"\n🔍 提取 {team_name} 玩家区域:")
                
                for i, region_config in enumerate(team_regions):
                    x = region_config.get('x', 0)
                    y = region_config.get('y', 0)
                    w = region_config.get('width', 100)
                    h = region_config.get('height', 50)
                    
                    # 检查坐标是否在图片范围内
                    if x + w > width or y + h > height:
                        print(f"  ⚠️  {region_config.get('name', f'player{i+1}')}: 坐标超出图片范围 ({x},{y},{w},{h})")
                        continue
                    
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
                        print(f"  ✅ {region_config.get('name', f'player{i+1}')}: ({x},{y}) {w}x{h}")
                    else:
                        print(f"  ❌ {region_config.get('name', f'player{i+1}')}: 区域为空")
            
        except Exception as e:
            print(f"❌ 提取玩家区域失败: {e}")
        
        return player_regions
    
    def test_image(self, image_path, save_debug=True, show_image=True):
        """测试单张图片"""
        print(f"🎯 开始测试图片: {image_path}")
        
        # 加载图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法加载图片: {image_path}")
            return None
        
        print(f"✅ 图片加载成功")
        
        # 提取玩家区域
        player_regions = self.extract_player_regions_from_image(image)
        
        total_players = len(player_regions['team1']) + len(player_regions['team2'])
        print(f"📊 提取到 {total_players} 个玩家区域 (Team1: {len(player_regions['team1'])}, Team2: {len(player_regions['team2'])})")
        
        if total_players == 0:
            print("❌ 未找到任何玩家区域，请检查配置文件中的坐标设置")
            return None
        
        # 分析玩家信息
        print("\n🔬 开始分析玩家信息...")
        results = self.analyzer.analyze_all_players(player_regions)
        
        # 显示结果
        self.display_results(results)
        
        # 保存调试图像
        if save_debug:
            self.save_debug_images(image, player_regions, results, image_path)
        
        # 显示图像
        if show_image:
            self.show_analysis_image(image, player_regions, results)
        
        return results
    
    def display_results(self, results):
        """显示分析结果"""
        print("\n" + "="*60)
        print("📋 检测结果")
        print("="*60)
        
        for team_name in ['team1', 'team2']:
            team_display = "队伍1" if team_name == 'team1' else "队伍2"
            print(f"\n🏆 {team_display}:")
            
            team_data = results.get(team_name, [])
            if not team_data:
                print("   无数据")
                continue
            
            for player in team_data:
                name = player.get('name', 'Unknown')
                hp = player.get('hp', 0)
                alive = player.get('alive', False)
                status = "🟢 存活" if alive else "🔴 死亡"
                coords = player.get('coords', (0, 0, 0, 0))
                
                print(f"   👤 {name}: {hp:3d}HP ({status}) - 区域: ({coords[0]},{coords[1]}) {coords[2]}x{coords[3]}")
        
        # 统计信息
        total_alive = sum(len([p for p in results.get(team, []) if p.get('alive', False)]) for team in ['team1', 'team2'])
        total_players = sum(len(results.get(team, [])) for team in ['team1', 'team2'])
        
        print(f"\n📊 统计: {total_alive}/{total_players} 玩家存活")
        print("="*60)
    
    def save_debug_images(self, original_image, player_regions, results, image_path):
        """保存调试图像"""
        try:
            debug_dir = Path("debug_output")
            debug_dir.mkdir(exist_ok=True)
            
            image_name = Path(image_path).stem
            
            # 保存原图副本
            cv2.imwrite(f"debug_output/{image_name}_original.png", original_image)
            
            # 保存每个玩家区域
            for team_name, team_players in player_regions.items():
                for i, player_data in enumerate(team_players):
                    region = player_data.get('region')
                    if region is not None and region.size > 0:
                        # 保存完整玩家区域
                        filename = f"debug_output/{image_name}_{team_name}_player{i+1}.png"
                        cv2.imwrite(filename, region)
                        
                        # 保存血量区域
                        hp_region = self.analyzer.extract_hp_region(region)
                        if hp_region is not None and hp_region.size > 0:
                            hp_filename = f"debug_output/{image_name}_{team_name}_player{i+1}_hp.png"
                            cv2.imwrite(hp_filename, hp_region)
                        
                        # 保存名称区域
                        name_region = self.analyzer.extract_name_region(region)
                        if name_region is not None and name_region.size > 0:
                            name_filename = f"debug_output/{image_name}_{team_name}_player{i+1}_name.png"
                            cv2.imwrite(name_filename, name_region)
            
            # 创建带标注的图像
            annotated_image = self.create_annotated_image(original_image, player_regions, results)
            cv2.imwrite(f"debug_output/{image_name}_annotated.png", annotated_image)
            
            # 创建分析结果图像
            debug_img = create_debug_image(player_regions, results)
            cv2.imwrite(f"debug_output/{image_name}_analysis.png", debug_img)
            
            print(f"💾 调试图像已保存到 debug_output/ 目录")
            
        except Exception as e:
            print(f"❌ 保存调试图像失败: {e}")
    
    def create_annotated_image(self, image, player_regions, results):
        """创建带标注的图像"""
        annotated = image.copy()
        
        try:
            # 为每个玩家区域绘制边框和信息
            for team_name, team_players in player_regions.items():
                color = (0, 255, 0) if team_name == 'team1' else (255, 0, 0)  # 绿色/红色
                
                for i, player_data in enumerate(team_players):
                    coords = player_data.get('coords', (0, 0, 0, 0))
                    x, y, w, h = coords
                    
                    # 绘制边框
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    
                    # 获取分析结果
                    team_results = results.get(team_name, [])
                    if i < len(team_results):
                        player_result = team_results[i]
                        hp = player_result.get('hp', 0)
                        alive = player_result.get('alive', False)
                        name = player_result.get('name', f'P{i+1}')
                        
                        # 绘制文本信息
                        status = "ALIVE" if alive else "DEAD"
                        # 如果名称不是默认的playerX格式，显示识别出的名称
                        display_name = name if not name.startswith('player') else f"P{i+1}"
                        text = f"{display_name}: {hp}HP ({status})"
                        
                        # 文本背景
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        cv2.rectangle(annotated, (x, y - 25), (x + text_size[0] + 4, y), color, -1)
                        
                        # 文本
                        text_color = (255, 255, 255)
                        cv2.putText(annotated, text, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                        
                        # 如果识别出了真实名称，在区域内部也显示
                        if not name.startswith('player') and name != "Unknown":
                            name_text = f"ID: {name}"
                            cv2.putText(annotated, name_text, (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
        except Exception as e:
            print(f"❌ 创建标注图像失败: {e}")
        
        return annotated
    
    def show_analysis_image(self, image, player_regions, results):
        """显示分析图像"""
        try:
            # 创建带标注的图像
            annotated = self.create_annotated_image(image, player_regions, results)
            
            # 缩放图像以适应屏幕
            height, width = annotated.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                annotated = cv2.resize(annotated, (new_width, new_height))
            
            # 显示图像
            cv2.imshow("CS Player Detection Test", annotated)
            print("\n👁️  图像显示中，按任意键关闭...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ 显示图像失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CS检测工具 - 图片测试')
    
    parser.add_argument('image_path', help='要测试的图片路径')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    parser.add_argument('--no-debug', action='store_true', help='不保存调试图像')
    parser.add_argument('--no-show', action='store_true', help='不显示图像')
    parser.add_argument('--save-json', help='保存结果到JSON文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ 图片文件不存在: {image_path}")
        sys.exit(1)
    
    # 创建测试器
    tester = ImageTester(args.config)
    
    # 执行测试
    results = tester.test_image(
        image_path,
        save_debug=not args.no_debug,
        show_image=not args.no_show
    )
    
    # 保存JSON结果
    if args.save_json and results:
        try:
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 结果已保存到: {args.save_json}")
        except Exception as e:
            print(f"❌ 保存JSON失败: {e}")


if __name__ == "__main__":
    main()