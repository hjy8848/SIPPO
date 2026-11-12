"""
ROSA SynthesizerAgent - 最终答案综合

继承 JADE 的 BaseLLMClient，复用 LLM 调用和 trace 机制。
核心逻辑来自 rosa_server_v2.py 的 FINAL_SYSTEM_MSG。
"""

import re
import logging
from typing import List, Dict, Optional, Any
from qa_manager.base_agent_tag import BaseLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================= Prompt 模板 =================

FINAL_SYSTEM_MSG = """Synthesize the resolved sub-questions to formulate a comprehensive and accurate final answer to the original user question.

**Instructions**:
1. Provide the final exact answer inside `<answer>...</answer>`.
2. The answer should be a concise phrase or name, NOT a full sentence, unless the original question explicitly asks for a description.
3. Only include the person's/entity's name when the question asks for a person/entity; do not prefix with phrases like "The answer is" or "X's spouse was".
4. If the answer is a date, location, or number, provide only that value."""


class SynthesizerAgent(BaseLLMClient):
    """
    最终答案综合 Agent
    
    继承 BaseLLMClient，复用 JADE 的 LLM 调用和 trace 机制。
    功能：
        - 基于已解决的子问题生成最终答案
        - 支持重试机制确保输出 <answer> 标签
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.3,
        max_retries: int = 1
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )
        self.max_retries = max_retries
        logging.info(f"初始化 SynthesizerAgent: model={model_name}")

    def _build_messages(
        self,
        original_question: str,
        resolved_sub_questions: str
    ) -> List[dict]:
        """
        构造综合答案的 prompt
        
        Args:
            original_question: 原始用户问题
            resolved_sub_questions: 已解决的子问题及其答案
        """
        return [
            {
                "role": "system",
                "content": FINAL_SYSTEM_MSG
            },
            {
                "role": "user",
                "content": (
                    f"**Original Question**: {original_question}\n\n"
                    f"**Resolved Sub-questions**:\n{resolved_sub_questions.strip()}"
                )
            }
        ]

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 响应，提取最终答案
        
        返回格式：
        {
            "is_valid": bool,
            "answer": str,
            "raw_output": str
        }
        """
        if not response or not response.strip():
            return {
                "is_valid": False,
                "answer": "",
                "raw_output": response,
                "error": "Empty response"
            }

        # 提取 <answer>
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else ""

        is_valid = bool(answer)

        return {
            "is_valid": is_valid,
            "answer": answer,
            "raw_output": response
        }

    async def synthesize(
        self,
        original_question: str,
        node_results: List[Dict[str, Any]],
        request_id: str = "default"
    ) -> Dict[str, Any]:
        """
        综合多个节点的执行结果生成最终答案
        
        Args:
            original_question: 原始用户问题
            node_results: 节点执行结果列表，每项包含 id, type, query, conclusion
            request_id: 请求 ID
            
        Returns:
            {
                "answer": str,
                "is_valid": bool,
                "raw_output": str
            }
        """
        # 构建已解决的子问题字符串
        history_summary = ""
        for node in node_results:
            history_summary += (
                f"Task {node.get('id', '?')} [{node.get('type', '?').upper()}]: "
                f"{node.get('query', '').strip()}\n"
                f"Conclusion: {node.get('conclusion', '')}\n\n"
            )

        messages = self._build_messages(original_question, history_summary)
        
        # 首次尝试
        response = await self.async_get_response(
            messages,
            agent_name="SynthesizerAgent",
            temperature=self.default_temperature
        )
        
        parsed = self.parse_response(response)

        # 重试机制
        retry_count = 0
        while not parsed["is_valid"] and retry_count < self.max_retries:
            logging.warning(f"[{request_id}] ⚠️ 答案格式无效，重试 {retry_count + 1}/{self.max_retries}")
            
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "Please output your final answer inside <answer>...</answer>."
            })
            
            response = await self.async_get_response(
                messages,
                agent_name="SynthesizerAgent",
                temperature=self.default_temperature
            )
            
            parsed = self.parse_response(response)
            retry_count += 1

        if not parsed["is_valid"]:
            logging.error(f"[{request_id}] ❌ 最终答案生成失败")

        logging.info(f"[{request_id}] 🎯 最终答案: {parsed.get('answer', 'N/A')}")

        return {
            "answer": parsed.get("answer", ""),
            "is_valid": parsed["is_valid"],
            "raw_output": parsed.get("raw_output", "")
        }


if __name__ == "__main__":
    import asyncio
    
    agent = SynthesizerAgent(model_name="gpt-4o")
    
    test_question = "What country was the performer of 'Shape of You' born in?"
    test_node_results = [
        {
            "id": "1",
            "type": "search",
            "query": "Who performs the song 'Shape of You'?",
            "conclusion": "Ed Sheeran"
        },
        {
            "id": "2",
            "type": "search",
            "query": "In which country was Ed Sheeran born?",
            "conclusion": "England"
        }
    ]
    
    async def test():
        print("=== 测试 SynthesizerAgent ===\n")
        result = await agent.synthesize(test_question, test_node_results, "test")
        print(f"原始问题: {test_question}")
        print(f"最终答案: {result['answer']}")
        print(f"有效: {result['is_valid']}")
    
    asyncio.run(test())
