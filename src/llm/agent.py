import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI
from llm.memory import AgentMemory
import torch


class Agent:
    def __init__(self, config_path, memory_folder="../intermediate_files/llms"):
        """
        初始化智能体模块。
        :param config_path: 配置文件路径
        :param memory_folder: 存储 memory.json 的文件夹路径
        """
        self.config = self.load_config(config_path)
        self.agent_id = self.config["agent_id"]
        self.system_prompt = self.config['system_message']
        self.model_type = self.config['training_setting']["model_type"]  # local / api
        self.api_key = self.config['training_setting']['api_key']
        self.model_path = self.config['training_setting']['model_path']
        self.client = None  # 如果是 API 模式，这里会被赋值为 OpenAI 客户端
        self.tokenizer = None  # 如果是本地模式，这里会被赋值为分词器
        self.model = None  # 如果是本地模式，这里会被赋值为模型

        # 初始化记忆模块
        self.memory = AgentMemory(self.config, memory_folder)

        # 根据模型类型初始化
        if self.model_type == "local":
            self.initialize_local_model()
        elif self.model_type == "api":
            self.initialize_api_client()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}. 请在配置文件中指定 'local' 或 'api'.")

    def load_config(self, path):
        """加载配置文件"""
        with open(path, 'r', encoding='utf-8') as file: # <--- 在这里添加 encoding='utf-8'
            return json.load(file)

    def initialize_local_model(self):
        """
        初始化本地模型和分词器。
        """
        try:
            if not self.model_path.strip():
                raise ValueError("本地模型路径为空，请在配置文件中设置 'model_path'。")

            print(f"正在加载本地模型：{self.model_path}")
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cuda",  # 自动分配 GPU
                torch_dtype=torch.bfloat16 if self.config['training_setting']['torch_dtype'] == "bfloat16" else torch.float16,
                trust_remote_code=True,  # 支持远程代码（针对某些自定义模型）
            )
            self.model.eval()
            print("本地模型加载完成。")

        except ImportError as e:
            raise ImportError(f"加载本地模型失败，请确保 transformers 已正确安装。错误信息: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"加载本地模型时发生错误：{str(e)}")

    def initialize_api_client(self):
        """
        初始化 OpenAI API 客户端。
        优先从配置读取 base_url，否则使用默认地址。
        """
        try:
            if not self.api_key.strip():
                raise ValueError("API 密钥为空，请在配置文件中设置 'api_key'。")
            # 支持从配置自定义 API 基础 URL（网页/代理地址）
            base_url = self.config.get("training_setting", {}).get("base_url")
            if not base_url:
                base_url = "https://api.zhizengzeng.com/v1"
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            # print("OpenAI API 客户端初始化完成。")
        except Exception as e:
            raise RuntimeError(f"初始化 OpenAI API 客户端时发生错误：{str(e)}")

    def generate_response(self, user_input, rag_response="", few_shot_examples=None):
        """
        生成响应。
        根据模型类型 (local/api) 调用不同的生成逻辑。
        :param user_input: 用户输入
        :param rag_response: 外部记忆的补充响应
        :return: AI 的响应
        """
        if self.model_type == "local":
            return self.generate_response_local(user_input, rag_response, few_shot_examples)
        elif self.model_type == "api":
            return self.generate_response_api(user_input, rag_response, few_shot_examples)

    def generate_response_local(self, user_input, rag_response=""):
        """
        使用本地模型生成响应。
        :param user_input: 用户输入
        :param rag_response: 外部记忆的补充响应
        :return: AI 的响应
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        if isinstance(rag_response, str):
            messages.append({"role": "assistant", "content": rag_response})
        else:
            messages.append({"role": "assistant", "content": "我无法检索到有效的响应。"})

        messages.append({"role": "user", "content": user_input})

        try:
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )

            generation_args = {
                "max_new_tokens": self.config['training_setting']['max_new_tokens'],
                "return_full_text": False,
                "temperature": self.config['training_setting']['temperature'],
                "do_sample": False,
            }

            output = pipe(messages, **generation_args)
            ai_response = output[0]['generated_text']

            # 保存到记忆模块
            self.memory.add_memory({"user": user_input, "response": ai_response})
            return ai_response

        except Exception as e:
            raise RuntimeError(f"本地模型生成响应时发生错误：{str(e)}")

    def generate_response_api(self, user_input, rag_response="", few_shot_examples=None):
        """
        使用 OpenAI API 生成响应。
        :param user_input: 用户输入
        :param rag_response: 外部记忆的补充响应
        :return: AI 的响应
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        # 注入Few-Shot范例
        if few_shot_examples and isinstance(few_shot_examples, list):
            messages.extend(few_shot_examples)
        
        if isinstance(rag_response, str):
            messages.append({"role": "assistant", "content": rag_response})
        else:
            messages.append({"role": "assistant", "content": "我无法检索到有效的响应。"})

        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.config['model'],
                max_tokens=self.config['training_setting']['max_new_tokens'],
                temperature=self.config['training_setting']['temperature'],
                top_p=self.config['training_setting']['top_p'],
            )
            ai_response = response.choices[0].message.content

            # 保存到记忆模块
            self.memory.add_memory({"user": user_input, "response": ai_response})
            return ai_response

        except Exception as e:
            raise RuntimeError(f"OpenAI API 生成响应时发生错误：{str(e)}")