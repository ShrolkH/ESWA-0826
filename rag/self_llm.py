from openai import OpenAI
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

GPT_API_KEY = ""
GPT_URL = ""
QWEN_API_KEY = ""
QWEN_URL = ""

class LLM_procket:
    """
    Encapsulates chat calls based on Alibaba Cloud DashScope OpenAI compatible interface.
    The default model is qwen-plus, and the API Key can be provided through the environment variable DASHSCOPE_API_KEY or explicitly passed as a parameter.
    """

    def __init__(self,
                model :str = "qwen-plus-2025-04-28",
                temperature = 0.7
                 ):
        if model in ['qwen-plus-latest','qwen2.5-32b-instruct',
                     'qwen-turbo-2025-04-28','qwen-plus-2025-04-28',
                     'qwen-max-2025-01-25','qwen2.5-72b-instruct']:
            # If not explicitly passed, read from environment variable
            self.api_key = QWEN_API_KEY 

            # Public cloud default address;
            self.base_url = QWEN_URL
        elif  model in ['gpt-4.1-2025-04-14','gpt-4o-2024-05-13','gpt-4',
                        'llama-3.1-405b','gpt-3.5-turbo','gpt-4.1','claude-3-7-sonnet-20250219',
                        'gemini-2.5-pro-exp-03-25','gpt-4o-mini'
                        ]:
            self.base_url = GPT_API_KEY
            self.api_key = GPT_API_KEY

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.model = model
        self.temperature = temperature

    def chat_with_llm(self, messages: list) -> str:
        """
        Call LLM for conversation. The messages must be in OpenAI Chat API format:
        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
            ...
        ]
        Returns the assistant's text content. If the request fails, save the input messages to a JSON file named with the current date in the East Eighth Time Zone.
        """
        try:
            # print(f'Current model being called: {self.model}, request URL: {self.base_url}')
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return completion.choices[0].message.content

        except Exception as err:
            # Get current date in East Eighth Time Zone
            tz = ZoneInfo("Asia/Shanghai")  # East Eighth Time Zone
            date_str = datetime.now(tz).strftime("%Y-%m-%d")
            # Construct file path
            filename = (
                "/hy-tmp/llm_lp-main/rag/result/"
                f"error_input_contain_inappropriate_content{date_str}.json"
            )
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # If the file exists, load existing content; otherwise, initialize an empty list
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    data = []
            else:
                data = []

            # Add current messages
            data.append({
                "timestamp": datetime.now(tz).isoformat(),
                "messages": messages
            })

            # Write back to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Return error message
            return f"Error: {err}. Request has been logged."