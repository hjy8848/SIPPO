"""
ROSA ReasonerAgent - 逻辑合并与计算

继承 JADE 的 BaseLLMClient，复用 LLM 调用和 trace 机制。
核心逻辑来自 rosa_server_v2.py 的 REASON_SYSTEM_MSG。
"""

import re
import logging
from typing import List, Dict, Optional, Any
from qa_manager.base_agent_tag import BaseLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================= Prompt 模板 =================

REASON_SYSTEM_MSG = """You are an expert Logical Analyzer. Resolve the logic task based ONLY on the provided task instructions.

**Instructions (Internal Reasoning - DO NOT OUTPUT)**:
1. Analyze the logic task step by step.
2. Perform the required computation, comparison, or synthesis.

**Output Format (ONLY output these tags, NO thinking blocks)**:
Provide the exact result inside `<conclusion>...</conclusion>`.

**IMPORTANT**: Do NOT output `<reason_think>` or any thinking block.

**EXAMPLE:**
Current Logic Task: Glenn Hughes was born on August 21, 1951 and Ross Lynch was born on December 29, 1995. Based on these dates, who is older?
Assistant:
<conclusion>Glenn Hughes</conclusion>"""


class ReasonerAgent(BaseLLMClient):
    """
    逻辑合并与计算 Agent
    
    继承 BaseLLMClient，复用 JADE 的 LLM 调用和 trace 机制。
    功能：
        - 合并/过滤/计算多个搜索结果
        - 执行逻辑推理任务
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.3
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )
        logging.info(f"初始化 ReasonerAgent: model={model_name}")

    def _build_messages(self, logic_task: str) -> List[dict]:
        """
        构造逻辑推理的 prompt
        
        Args:
            logic_task: 逻辑任务描述（包含 #ID# 变量引用）
        """
        return [
            {
                "role": "system",
                "content": REASON_SYSTEM_MSG
            },
            {
                "role": "user",
                "content": f"**Current Logic Task**: {logic_task}"
            }
        ]

    def parse_response(self, response: str, original_task: str) -> Dict[str, Any]:
        """
        解析 LLM 响应，提取推理结果
        
        返回格式：
        {
            "is_valid": bool,
            "conclusion": str,
            "raw_output": str
        }
        """
        if not response or not response.strip():
            return {
                "is_valid": False,
                "conclusion": "",
                "raw_output": response,
                "error": "Empty response"
            }

        # 提取 <conclusion>
        conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', response, re.DOTALL | re.IGNORECASE)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

        # 校验
        is_valid = bool(conclusion) and len(conclusion) > 0

        return {
            "is_valid": is_valid,
            "conclusion": conclusion if is_valid else original_task,
            "raw_output": response
        }

    def reason(self, logic_task: str) -> Dict[str, Any]:
        """
        同步执行逻辑推理
        """
        messages = self._build_messages(logic_task)
        response = self.get_response(messages, temperature=self.default_temperature)
        return self.parse_response(response, logic_task)

    async def reason_async(self, logic_task: str) -> Dict[str, Any]:
        """
        异步执行逻辑推理
        """
        messages = self._build_messages(logic_task)
        response = await self.async_get_response(
            messages,
            agent_name="ReasonerAgent",
            temperature=self.default_temperature
        )
        return self.parse_response(response, logic_task)

    async def reason_parallel(self, logic_tasks: List[str]) -> List[Dict[str, Any]]:
        """
        并行执行多个逻辑推理任务
        """
        batch_messages = [self._build_messages(task) for task in logic_tasks]
        responses = await self.async_batch_get_response(
            batch_messages,
            agent_name="ReasonerAgent",
            temperature=self.default_temperature
        )
        return [
            self.parse_response(resp, logic_tasks[i])
            for i, resp in enumerate(responses)
        ]


if __name__ == "__main__":
    import asyncio
    
    agent = ReasonerAgent(model_name="gpt-4o")
    
    test_tasks = [
        "Person 1 is Glenn Hughes (born August 21, 1951). Person 2 is Ross Lynch (born December 29, 1995). Output the exact names of these two actors so we can search for their co-starred movies.",
        "Person 1 is Robert Downey Jr. Person 2 is Scarlett Johansson. Output the exact names of these two actors so we can search for their co-starred movies."
    ]
    
    async def test():
        print("=== 测试 ReasonerAgent ===\n")
        results = await agent.reason_parallel(test_tasks)
        for i, result in enumerate(results):
            print(f"任务 {i+1}: {test_tasks[i][:50]}...")
            print(f"结论: {result['conclusion']}")
            print(f"有效: {result['is_valid']}")
            print("-" * 50)
    
    asyncio.run(test())
