"""
ROSA QueryRewriterAgent - RT-RAG 语义重构

继承 JADE 的 BaseLLMClient，复用 LLM 调用和 trace 机制。
核心逻辑来自 rosa_server_v2.py 的 QUERY_REWRITE_SYSTEM_MSG。
"""

import re
import logging
from typing import List, Dict, Optional, Any
from qa_manager.base_agent_tag import BaseLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================= Prompt 模板 =================

QUERY_REWRITE_SYSTEM_MSG = """You are a Search Query Optimizer. Transform draft search queries into clean, natural, retrieval-friendly questions.

RULES:
1. Replace all #X# placeholders with their actual values naturally.
2. Remove redundant information from the substitution (e.g., remove duplicate entity mentions like "Ocala, Florida" → just "Ocala").
3. Make the query a complete, standalone question.
4. Keep it concise and entity-rich for vector search.
5. Do NOT answer the question — only rewrite the query.
6. Output ONLY the rewritten query, nothing else. No tags, no explanation.
7. Preserve all specific constraints from the Original Goal (dates, locations, years, etc.) in the rewritten query.

EXAMPLES:
Original Goal: In what part of Florida is Tom Denney's birthplace located?
Draft: In which part of Florida is #1# located?
#1# = Ocala, Florida
Output: What part of Florida is Ocala in?

Original Goal: What century did the author of A Treatise Concerning the Principles of Human Knowledge live in?
Draft: In which century did the author of #1# live?
#1# = George Berkeley
Output: In which century did George Berkeley live?

Original Goal: In what city did the inventor of the light bulb live in 1880?
Draft: In which city did #1# live?
#1# = Thomas Edison
Output: In which city did Thomas Edison live in 1880?

Original Goal: What experimental satellite was the forerunner to the communication satellite of the operator of GSAT-7?
Draft: What experimental satellite was the forerunner to the communication satellite operated by #1#?
#1# = Indian Air Force
Output: What experimental satellite was the forerunner to the Indian Air Force communication satellite?"""


class QueryRewriterAgent(BaseLLMClient):
    """
    RT-RAG 语义重构 Agent
    
    继承 BaseLLMClient，复用 JADE 的 LLM 调用和 trace 机制。
    功能：
        - 将带 #ID# 占位符的查询转换为可检索的自然语言查询
        - 保持原始约束信息（日期、地点等）
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
        logging.info(f"初始化 QueryRewriterAgent: model={model_name}")

    def _build_messages(
        self,
        original_goal: str,
        draft_query: str,
        resolved_context: Dict[str, str]
    ) -> List[dict]:
        """
        构造语义重构的 prompt
        
        Args:
            original_goal: 原始用户问题
            draft_query: 带 #ID# 占位符的草稿查询
            resolved_context: 已解析的上下文，格式 {id: value, ...}
        """
        # 构建变量替换信息
        context_lines = []
        for var_id, value in sorted(resolved_context.items(), key=lambda x: int(x[0])):
            context_lines.append(f"#{var_id}# = {value}")
        context_str = "\n".join(context_lines)

        return [
            {
                "role": "system",
                "content": QUERY_REWRITE_SYSTEM_MSG
            },
            {
                "role": "user",
                "content": (
                    f"Original Goal: {original_goal}\n"
                    f"Draft: {draft_query}\n"
                    f"{context_str}\n"
                    f"Rewritten query:"
                )
            }
        ]

    def parse_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """
        解析 LLM 响应，提取重构后的查询
        
        返回格式：
        {
            "is_valid": bool,
            "rewritten_query": str,
            "raw_output": str
        }
        """
        if not response or not response.strip():
            return {
                "is_valid": False,
                "rewritten_query": original_query,
                "raw_output": response,
                "error": "Empty response"
            }

        # 清理响应，取第一行作为查询
        rewritten = response.strip().split('\n')[0].strip()
        
        # 校验：不能为空且长度合理
        is_valid = (
            len(rewritten) > 3 and
            len(rewritten) <= 256 and
            "#" not in rewritten  # 不应包含未替换的占位符
        )

        return {
            "is_valid": is_valid,
            "rewritten_query": rewritten if is_valid else original_query,
            "raw_output": response
        }

    def rewrite(
        self,
        original_goal: str,
        draft_query: str,
        resolved_context: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        同步重构单个查询
        """
        messages = self._build_messages(original_goal, draft_query, resolved_context)
        response = self.get_response(messages, temperature=self.default_temperature)
        return self.parse_response(response, draft_query)

    async def rewrite_async(
        self,
        original_goal: str,
        draft_query: str,
        resolved_context: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        异步重构单个查询
        """
        messages = self._build_messages(original_goal, draft_query, resolved_context)
        response = await self.async_get_response(
            messages,
            agent_name="QueryRewriterAgent",
            temperature=self.default_temperature
        )
        return self.parse_response(response, draft_query)

    async def rewrite_parallel(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        并行重构多个查询
        
        Args:
            items: List[{
                "original_goal": str,
                "draft_query": str,
                "resolved_context": Dict[str, str]
            }]
        """
        batch_messages = [
            self._build_messages(item["original_goal"], item["draft_query"], item["resolved_context"])
            for item in items
        ]
        responses = await self.async_batch_get_response(
            batch_messages,
            agent_name="QueryRewriterAgent",
            temperature=self.default_temperature
        )
        return [
            self.parse_response(resp, items[i]["draft_query"])
            for i, resp in enumerate(responses)
        ]


if __name__ == "__main__":
    import asyncio
    
    agent = QueryRewriterAgent(model_name="gpt-4o")
    
    test_items = [
        {
            "original_goal": "What country was the performer of 'Shape of You' born in?",
            "draft_query": "In which country was #1# born?",
            "resolved_context": {"1": "Ed Sheeran"}
        },
        {
            "original_goal": "What movie features both the actor who played Iron Man in 2008 and the actress who played Black Widow?",
            "draft_query": "What movie features both actors identified in #3# in its cast?",
            "resolved_context": {
                "1": "Robert Downey Jr.",
                "2": "Scarlett Johansson",
                "3": "Robert Downey Jr. and Scarlett Johansson"
            }
        }
    ]
    
    async def test():
        print("=== 测试 QueryRewriterAgent ===\n")
        results = await agent.rewrite_parallel(test_items)
        for i, result in enumerate(results):
            print(f"测试 {i+1}:")
            print(f"原始: {test_items[i]['draft_query']}")
            print(f"重写: {result['rewritten_query']}")
            print(f"有效: {result['is_valid']}")
            print("-" * 50)
    
    asyncio.run(test())
