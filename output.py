#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS直播画面检测结果输出模块
负责将检测结果输出到控制台、文件或HTTP API
"""

import json
import logging
import time
from typing import Dict, Any
from pathlib import Path
import threading
from flask import Flask, jsonify, request
from flask.logging import default_handler

class OutputManager:
    """结果输出管理器"""
    
    def __init__(self, config):
        """
        初始化输出管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.output_config = config.get('output', {})
        
        # 输出选项
        self.console_output = self.output_config.get('console', True)
        self.file_output = self.output_config.get('file', True)
        self.file_path = self.output_config.get('file_path', 'output.json')
        self.http_api = self.output_config.get('http_api', False)
        self.api_port = self.output_config.get('api_port', 8080)
        
        # 最新数据存储
        self.latest_data = {}
        self.data_lock = threading.Lock()
        
        # HTTP API服务器
        self.app = None
        self.server_thread = None
        
        if self.http_api:
            self._setup_http_api()
        
        logging.info("输出管理器初始化完成")
    
    def _setup_http_api(self):
        """设置HTTP API服务器"""
        try:
            self.app = Flask(__name__)
            
            # 禁用Flask的默认日志输出
            self.app.logger.removeHandler(default_handler)
            
            @self.app.route('/api/players', methods=['GET'])
            def get_players():
                """获取玩家信息API"""
                with self.data_lock:
                    return jsonify(self.latest_data)
            
            @self.app.route('/api/status', methods=['GET'])
            def get_status():
                """获取服务状态API"""
                return jsonify({
                    'status': 'running',
                    'timestamp': int(time.time() * 1000),
                    'has_data': bool(self.latest_data)
                })
            
            @self.app.route('/api/config', methods=['GET'])
            def get_config():
                """获取配置信息API"""
                return jsonify({
                    'detection_fps': self.config.get('detection', {}).get('fps', 5),
                    'player_count': {
                        'team1': len(self.config.get('player_regions', {}).get('team1', [])),
                        'team2': len(self.config.get('player_regions', {}).get('team2', []))
                    }
                })
            
            logging.info(f"HTTP API服务器设置完成，端口: {self.api_port}")
            
        except Exception as e:
            logging.error(f"设置HTTP API失败: {e}")
            self.http_api = False
    
    def start_http_server(self):
        """启动HTTP服务器"""
        if not self.http_api or not self.app:
            return
        
        try:
            def run_server():
                self.app.run(host='0.0.0.0', port=self.api_port, debug=False, use_reloader=False)
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            logging.info(f"HTTP API服务器已启动: http://localhost:{self.api_port}")
            
        except Exception as e:
            logging.error(f"启动HTTP服务器失败: {e}")
    
    def output_console(self, data: Dict[str, Any]):
        """输出到控制台"""
        if not self.console_output:
            return
        
        try:
            # 格式化输出
            print("\n" + "="*50)
            print(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-"*50)
            
            for team_name in ['team1', 'team2']:
                team_data = data.get(team_name, [])
                print(f"\n{team_name.upper()}:")
                
                if not team_data:
                    print("  无数据")
                    continue
                
                for player in team_data:
                    status = "存活" if player.get('alive', False) else "死亡"
                    hp = player.get('hp', 0)
                    name = player.get('name', 'Unknown')
                    
                    print(f"  {name}: 血量={hp:3d} ({status})")
            
            print("="*50)
            
        except Exception as e:
            logging.error(f"控制台输出失败: {e}")
    
    def output_file(self, data: Dict[str, Any]):
        """输出到文件"""
        if not self.file_output:
            return
        
        try:
            # 确保目录存在
            file_path = Path(self.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 添加时间戳
            output_data = {
                **data,
                'detection_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp_ms': int(time.time() * 1000)
            }
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logging.debug(f"结果已写入文件: {file_path}")
            
        except Exception as e:
            logging.error(f"文件输出失败: {e}")
    
    def update_api_data(self, data: Dict[str, Any]):
        """更新API数据"""
        if not self.http_api:
            return
        
        try:
            with self.data_lock:
                self.latest_data = {
                    **data,
                    'detection_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp_ms': int(time.time() * 1000)
                }
            
            logging.debug("API数据已更新")
            
        except Exception as e:
            logging.error(f"更新API数据失败: {e}")
    
    def output_results(self, data: Dict[str, Any]):
        """
        输出检测结果到所有配置的目标
        
        Args:
            data: 检测结果数据
        """
        if not data:
            return
        
        try:
            # 控制台输出
            self.output_console(data)
            
            # 文件输出
            self.output_file(data)
            
            # HTTP API数据更新
            self.update_api_data(data)
            
        except Exception as e:
            logging.error(f"输出结果失败: {e}")
    
    def get_summary_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取统计摘要
        
        Args:
            data: 检测结果数据
            
        Returns:
            统计摘要
        """
        try:
            stats = {
                'total_players': 0,
                'alive_players': 0,
                'dead_players': 0,
                'team1_alive': 0,
                'team2_alive': 0,
                'average_hp': 0
            }
            
            total_hp = 0
            
            for team_name in ['team1', 'team2']:
                team_data = data.get(team_name, [])
                team_alive = 0
                
                for player in team_data:
                    stats['total_players'] += 1
                    
                    if player.get('alive', False):
                        stats['alive_players'] += 1
                        team_alive += 1
                    else:
                        stats['dead_players'] += 1
                    
                    total_hp += player.get('hp', 0)
                
                if team_name == 'team1':
                    stats['team1_alive'] = team_alive
                else:
                    stats['team2_alive'] = team_alive
            
            if stats['total_players'] > 0:
                stats['average_hp'] = total_hp / stats['total_players']
            
            return stats
            
        except Exception as e:
            logging.error(f"计算统计摘要失败: {e}")
            return {}
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.server_thread and self.server_thread.is_alive():
                # Flask服务器无法优雅关闭，但这里可以记录日志
                logging.info("HTTP API服务器线程正在清理...")
            
            logging.info("输出管理器资源清理完成")
            
        except Exception as e:
            logging.error(f"清理输出管理器资源失败: {e}")


class ResultFormatter:
    """结果格式化器"""
    
    @staticmethod
    def format_for_display(data: Dict[str, Any]) -> str:
        """
        格式化结果用于显示
        
        Args:
            data: 原始数据
            
        Returns:
            格式化后的字符串
        """
        lines = []
        
        try:
            lines.append("CS玩家状态检测结果")
            lines.append("=" * 30)
            
            for team_name in ['team1', 'team2']:
                team_display = "队伍1" if team_name == 'team1' else "队伍2"
                lines.append(f"\n{team_display}:")
                
                team_data = data.get(team_name, [])
                if not team_data:
                    lines.append("  无数据")
                    continue
                
                for i, player in enumerate(team_data):
                    name = player.get('name', f'玩家{i+1}')
                    hp = player.get('hp', 0)
                    alive = player.get('alive', False)
                    status = "存活" if alive else "死亡"
                    
                    lines.append(f"  {name}: {hp:3d}HP ({status})")
            
            return "\n".join(lines)
            
        except Exception as e:
            logging.error(f"格式化显示结果失败: {e}")
            return "格式化失败"
    
    @staticmethod
    def format_for_json(data: Dict[str, Any], pretty: bool = True) -> str:
        """
        格式化结果为JSON
        
        Args:
            data: 原始数据
            pretty: 是否美化输出
            
        Returns:
            JSON字符串
        """
        try:
            if pretty:
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                return json.dumps(data, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"格式化JSON结果失败: {e}")
            return "{}"


if __name__ == "__main__":
    # 测试代码
    import json
    
    # 加载配置
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建输出管理器
    output_manager = OutputManager(config)
    
    # 启动HTTP服务器
    output_manager.start_http_server()
    
    # 模拟测试数据
    test_data = {
        'team1': [
            {'name': 'Player1', 'hp': 100, 'alive': True},
            {'name': 'Player2', 'hp': 75, 'alive': True},
            {'name': 'Player3', 'hp': 0, 'alive': False},
        ],
        'team2': [
            {'name': 'Player4', 'hp': 50, 'alive': True},
            {'name': 'Player5', 'hp': 25, 'alive': True},
        ]
    }
    
    # 测试输出
    print("测试输出功能...")
    output_manager.output_results(test_data)
    
    # 显示统计
    stats = output_manager.get_summary_stats(test_data)
    print(f"\n统计摘要: {stats}")
    
    if config.get('output', {}).get('http_api', False):
        print(f"\nHTTP API已启动，访问: http://localhost:{output_manager.api_port}/api/players")
        print("按回车键退出...")
        input()
    
    output_manager.cleanup()