"""
ROSA SearcherAgent - 外部检索与答案提取

继承 JADE 的 BaseLLMClient，复用 LLM 调用和 trace 机制。
核心逻辑来自 rosa_server_v2.py 的 SEARCH_SYSTEM_MSG 和 fetch_search_results。
"""

import re
import logging
import httpx
from typing import List, Dict, Optional, Any
from qa_manager.base_agent_tag import BaseLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================= Prompt 模板 =================

SEARCH_SYSTEM_MSG = """You are an expert Evidence Evaluator. Resolve the Current Sub-question based ONLY on the Search Context.

**Instructions (Internal Reasoning - DO NOT OUTPUT)**:
1. Extract relevant info from the Search Context.
2. Check sufficiency of the information.
3. Constraint Check: Identify ANY constraints in the Sub-question (e.g., "first", "second", "most", "before", "after", dates, locations). Apply them to select the correct answer.
4. Validate entity type.

**Output Format (ONLY output these tags, NO thinking blocks)**:
1. Output a `<ref>` tag citing one evidence doc id, e.g., `<ref>[1_1]</ref>`.
2. Output the exact, concise answer inside `<conclusion>...</conclusion>`.

**IMPORTANT**:
- Do NOT output `<search_think>` or any thinking block.
- You MUST always output a `<conclusion>`. Never output placeholder answers.
- If the context does not contain enough information to answer, try to reformulate the question into a better retrieval query inside `<rewrite_query>...</rewrite_query>`.
- You can request at most two rewrites for the same sub-question.
- Do NOT output both `<conclusion>` and `<rewrite_query>` at the same time.

**EXAMPLES:**

Search Context: [Doc_1]: John was married three times: to Alice (1980-1985), Mary (1986-1990), and Sarah (1991-present).
Current Sub-question: Who was the second wife of John?
Assistant:
<ref>[1_1]</ref>
<conclusion>Mary</conclusion>

Search Context: [Doc_1]: In 2014, Microsoft announced a deal to buy the Swedish studio Mojang.
Current Sub-question: Which organization acquired Mojang?
Assistant:
<ref>[1_1]</ref>
<conclusion>Microsoft</conclusion>

Search Context: [Doc_1]: Roman emperor Augustus ruled from 27 BC to AD 14. Tiberius succeeded Augustus and ruled from AD 14 to 37.
Current Sub-question: Who succeeded the longest reigning Roman emperor?
Assistant:
<ref>[1_1]</ref>
<conclusion>Tiberius</conclusion>"""


class SearcherAgent(BaseLLMClient):
    """
    外部检索与答案提取 Agent
    
    继承 BaseLLMClient，复用 JADE 的 LLM 调用和 trace 机制。
    功能：
        - 调用远程检索 API 获取相关文档
        - 基于检索上下文提取答案
        - 支持 rewrite 重试机制
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0,
        retriever_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 3,
        max_retries: int = 2
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )
        self.retriever_url = retriever_url
        self.topk = topk
        self.max_retries = max_retries
        logging.info(f"初始化 SearcherAgent: model={model_name}, retriever_url={retriever_url}")

    async def fetch_search_results(self, query: str) -> str:
        """
        调用远程检索 API 获取文档
        
        Args:
            query: 检索查询
            
        Returns:
            格式化后的文档字符串
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.retriever_url,
                    json={"queries": [query], "topk": self.topk, "return_scores": True}
                )
                resp.raise_for_status()
                results = resp.json().get('result', [[]])[0]

            res_str = ''
            for idx, doc in enumerate(results):
                content = str(doc.get('document', {}).get('contents', ''))
                title = content.split("\n")[0] if "\n" in content else "Unknown"
                res_str += f"[1_{idx+1}]: (Title: {title}) {content}\n"

            return res_str.strip() if res_str else f"[No Documents Found] for '{query}'."
        except Exception as e:
            logging.warning(f"检索 API 错误: {e}")
            return f"[Search API Error]: {e}"

    def _build_messages(self, query: str, search_context: str) -> List[dict]:
        """
        构造答案提取的 prompt
        
        Args:
            query: 当前子问题
            search_context: 检索到的文档上下文
        """
        return [
            {
                "role": "system",
                "content": SEARCH_SYSTEM_MSG
            },
            {
                "role": "user",
                "content": (
                    f"**Current Sub-question**: {query}\n\n"
                    f"**Search Context**:\n{search_context}"
                )
            }
        ]

    def parse_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """
        解析 LLM 响应，提取答案或重写查询
        
        返回格式：
        {
            "is_valid": bool,
            "has_conclusion": bool,
            "conclusion": str,
            "rewrite_query": str,      # 如果需要重试
            "raw_output": str
        }
        """
        if not response or not response.strip():
            return {
                "is_valid": False,
                "has_conclusion": False,
                "conclusion": "",
                "rewrite_query": "",
                "raw_output": response,
                "error": "Empty response"
            }

        # 提取 <conclusion>
        conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', response, re.DOTALL | re.IGNORECASE)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

        # 提取 <rewrite_query>
        rewrite_match = re.search(r'<rewrite_query>(.*?)</rewrite_query>', response, re.DOTALL | re.IGNORECASE)
        rewrite_query = rewrite_match.group(1).strip() if rewrite_match else ""

        has_conclusion = bool(conclusion)
        needs_rewrite = bool(rewrite_query and not conclusion)

        return {
            "is_valid": has_conclusion or needs_rewrite,
            "has_conclusion": has_conclusion,
            "conclusion": conclusion,
            "rewrite_query": rewrite_query,
            "raw_output": response
        }

    async def search(
        self,
        query: str,
        request_id: str = "default"
    ) -> Dict[str, Any]:
        """
        执行单个搜索：检索 + 答案提取 + 重试
        
        Args:
            query: 搜索查询
            request_id: 请求 ID，用于日志
            
        Returns:
            {
                "query": str,           # 最终使用的查询
                "search_context": str,   # 检索到的文档
                "conclusion": str,      # 提取的答案
                "raw_output": str,
                "num_retries": int
            }
        """
        current_query = query
        search_context = ""
        final_conclusion = ""
        num_retries = 0
        full_response = ""

        for retry in range(self.max_retries + 1):
            # 检索
            search_context = await self.fetch_search_results(current_query)
            logging.info(f"[{request_id}] 🔍 检索: '{current_query}'")

            # 答案提取
            messages = self._build_messages(current_query, search_context)
            response = await self.async_get_response(
                messages,
                agent_name="SearcherAgent",
                temperature=self.default_temperature
            )
            full_response = response
            num_retries = retry

            # 解析
            parsed = self.parse_response(response, current_query)

            if parsed["has_conclusion"]:
                final_conclusion = parsed["conclusion"]
                logging.info(f"[{request_id}] ✅ 答案: {final_conclusion}")
                break
            elif parsed["rewrite_query"]:
                # 需要重试
                old_query = current_query
                current_query = parsed["rewrite_query"].strip()
                logging.info(f"[{request_id}] 🔄 重写: '{old_query}' → '{current_query}'")
                continue
            else:
                # 无法提取答案，用原始查询作为保底
                final_conclusion = current_query
                logging.warning(f"[{request_id}] ⚠️ 无法提取答案，使用查询作为答案")
                break

        return {
            "query": current_query,
            "search_context": search_context,
            "conclusion": final_conclusion,
            "raw_output": full_response,
            "num_retries": num_retries
        }

    async def search_parallel(
        self,
        queries: List[str],
        request_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        并行执行多个搜索
        """
        import asyncio
        tasks = [self.search(q, f"{request_id}-{i}") for i, q in enumerate(queries)]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    import asyncio
    
    # 注意：需要配置正确的 retriever_url
    agent = SearcherAgent(
        model_name="gpt-4o",
        retriever_url="http://127.0.0.1:8000/retrieve"
    )
    
    test_queries = [
        "Who performs the song Shape of You?",
        "What is the birth date of Glenn Hughes?"
    ]
    
    async def test():
        print("=== 测试 SearcherAgent ===\n")
        results = await agent.search_parallel(test_queries, "test")
        for i, result in enumerate(results):
            print(f"查询 {i+1}: {test_queries[i]}")
            print(f"答案: {result['conclusion']}")
            print(f"重试次数: {result['num_retries']}")
            print("-" * 50)
    
    asyncio.run(test())
