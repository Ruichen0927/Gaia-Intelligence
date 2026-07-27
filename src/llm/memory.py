import os
import json
from datetime import datetime


class AgentMemory:
    def __init__(self, config, base_folder="../intermediate_files/llms"):
        """
        初始化智能体记忆模块。
        :param config: 配置文件内容，包含智能体的初始配置信息
        :param base_folder: 存储 memory.json 的基础文件夹路径
        """
        self.agent_id = config["agent_id"]
        self.base_folder = base_folder
        self.memory_folder = os.path.join(base_folder, self.agent_id)
        self.file_path = os.path.join(self.memory_folder, "memory.json")
        self.memory_data = self._initialize_memory()

    def _initialize_memory(self):
        """
        初始化 memory.json 文件。如果文件夹或文件不存在，则创建它们并保存一个空的初始结构。
        :return: 初始化后的 memory 数据
        """
        # 确保文件夹存在
        os.makedirs(self.memory_folder, exist_ok=True)

        # 如果 memory.json 已存在，加载其内容
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 如果文件不存在，创建一个空的 memory 数据结构
        memory_data = {
            "agent_id": self.agent_id,
            "memory": [],  # 空记忆列表
            "metadata": {
                "capacity": 100  # 默认容量限制
            }
        }

        # 保存初始化的空 memory.json
        with open(self.file_path, "w") as f:
            json.dump(memory_data, f, indent=4)

        return memory_data

    def _save_memory(self):
        """将内存中的数据保存到 memory.json 文件，确保中文字符不被转义。"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_data, f, indent=4, ensure_ascii=False)

    def add_memory(self, data):
        """
        添加一条新的记忆到 memory 中。
        :param data: 记忆内容，字符串或嵌套结构
        """
        new_memory = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.memory_data["memory"].append(new_memory)

        # 检查是否超过容量限制
        capacity = self.memory_data["metadata"].get("capacity", 100)
        if len(self.memory_data["memory"]) > capacity:
            self.memory_data["memory"].pop(0)  # 删除最早的记忆

        # 保存更新后的记忆
        self._save_memory()

    def query_memory(self, keyword):
        """
        按关键词搜索记忆内容。
        :param keyword: 要查询的关键词
        :return: 包含关键词的记忆列表
        """
        return [
            memory for memory in self.memory_data["memory"]
            if keyword in memory["data"]
        ]

    def get_recent_memory(self, n=5):
        """
        获取最近的 n 条记忆。
        :param n: 返回的记忆条数
        :return: 最近 n 条记忆的列表
        """
        return self.memory_data["memory"][-n:]

    def get_all_memory(self):
        """
        获取所有记忆。
        :return: 内存中所有记忆的列表
        """
        return self.memory_data["memory"]

    def clear_memory(self):
        """清空所有记忆。"""
        self.memory_data["memory"] = []
        self._save_memory()

class SharedMemory:
    def __init__(self, memory_folder="../intermediate_files/llms"):
        """
        初始化共享记忆模块。
        :param memory_folder: 存储 share_memory.json 的文件夹路径
        """
        self.memory_folder = memory_folder
        self.file_path = os.path.join(memory_folder, "share_memory.json")
        self.memory_data = self._load_memory()

    def _load_memory(self):
        """加载 share_memory.json 文件，如果不存在则初始化为空结构。"""
        if not os.path.exists(self.memory_folder):
            os.makedirs(self.memory_folder)
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # 初始化空的共享记忆数据结构
        return {"memory": []}

    def _save_memory(self):
        """将内存中的数据保存到 memory.json 文件，确保中文字符不被转义。"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_data, f, indent=4, ensure_ascii=False)

    def add_memory(self, data):
        """
        添加一条新的共享记忆。
        :param data: 记忆内容，字符串或嵌套结构
        """
        new_memory = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.memory_data["memory"].append(new_memory)
        self._save_memory()

    def get_all_memory(self):
        """
        获取所有共享记忆。
        :return: 共享记忆的列表
        """
        return self.memory_data["memory"]

    def clear_memory(self):
        """清空所有共享记忆。"""
        self.memory_data["memory"] = []
        self._save_memory()