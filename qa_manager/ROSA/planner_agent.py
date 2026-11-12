"""
ROSA PlannerAgent - DAG 规划与问题分类

继承 JADE 的 BaseLLMClient，复用 LLM 调用和 trace 机制。
核心逻辑来自 rosa_server_v2.py 的 PLANNING_SYSTEM_MSG。
"""

import re
import logging
from typing import List, Dict, Optional, Any
from qa_manager.base_agent_tag import BaseLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================= Prompt 模板 =================

PLANNING_SYSTEM_MSG = """You are an expert logical Graph Architect. Your task is to break down complex multi-hop questions into a precise Directed Acyclic Graph (DAG) of atomic execution nodes.
Do not guess or assume answers. You are building the blueprint for a downstream execution engine.

**1. CLASSIFY FIRST, THEN DECOMPOSE**
Follow the decision tree below to determine the question type and its decomposition strategy.

---
### How to classify: Answer these questions in order

**Q1: Does the next step ALWAYS wait for the previous step's answer?**
- If YES for every hop → `chain`
- If NO, go to Q2

**Q2: Can ALL sub-searches start simultaneously (they are independent)?**
- If YES, go to Q3
- If NO (some searches depend on others) → `hybrid`

**Q3: Does the final answer come directly from comparing/combining the search results?**
- If YES → `parallel`
- If NO (need one more search after merging results) → `hybrid`

---

### Type-specific decomposition rules

#### `chain`: Strict sequential search
1. Identify the first unknown fact in the sequence.
2. Search it.
3. Use the answer as input to the next search.
4. Repeat until the target is reached.
5. **Rule**: ALWAYS use `#ID#` to link steps. The answer of step N feeds directly into step N+1.

#### `parallel`: Concurrent independent search
1. Identify ALL independent facts needed (they are all named or inferable from the question).
2. Search them ALL at once — no step waits for another.
3. Compare or combine the results to produce the final answer.
4. **Rule**: DO NOT use `#ID#` between parallel branches. Each branch stands alone.

#### `hybrid`: Parallel-then-merge search
1. Identify independent branches that can start simultaneously.
2. Execute those branches in parallel.
3. The results of those branches MUST be merged/synthesized into a NEW query.
4. That new query is the final search step.
5. **Rule**: Use `#ID#` only at the merge point. Parallel branches have no `#ID#` between them, but the merge step uses `#ID#` to reference branch results.

---

**2. NODE DEFINITIONS & RULES**
- `<search id="X">...</search>`: For retrieving unknown facts from external knowledge.
    * RULE 1 (Atomic): Ask for ONE fact only per search.
    * RULE 2 (Parallelism): If multiple searches do NOT depend on each other, do NOT use `#ID#` between them. This allows the engine to run them simultaneously.
- `<reason id="X">...</reason>`: For intermediate logic or data synthesis.
    * RULE 3 (The Golden Rule of Reasoning): Use a `<reason>` node ONLY when you must merge, filter, or compute data from previous searches to enable a SUBSEQUENT `<search>`. If there are no downstream searches left, DO NOT use a `<reason>` node. Let the graph simply end with `<search>`.
    * RULE 4 (The Blind Worker Rule): Think of the `<reason>` node as a prompt you are writing for a separate, blind AI assistant. It does NOT know the original user question. You MUST follow this formula: [State facts using #ID#] + [Ask an explicit question]. NEVER write declarative conclusions (e.g., "Therefore, the answer is #1#.").
        BAD: <reason id="3">Extract their names.</reason>
        GOOD: <reason id="3">Person 1 is #1#. Person 2 is #2#. Output the exact names of these two actors so we can search for their co-starred movies.</reason>

**3. DECOMPOSITION METHODOLOGY (THINK BEFORE ACT)**
To correctly map the graph, you MUST follow this deductive process inside your `<plan_think>...</plan_think>` block:
* **Step 1 - Entity Analysis**: List the explicit entities given in the question and the final Target.
* **Step 2 - Missing Link Analysis**: What unknown facts act as "stepping stones" between Given Entities and the Target? Are any steps dependent on prior answers?
* **Step 3 - Classify via Q1/Q2/Q3**: Use the decision tree above. State which question you answered and what type you chose.
* **Step 4 - Node Drafting**: Draft the search/reason steps following the type-specific rules above. Actively manage the `#ID#` variables.

After thinking, output exactly one `<plan>...</plan>` block containing the nodes.
Your `<plan>` MUST start with a `<type>` tag declaring your classification:
- `<type>chain</type>` OR `<type>parallel</type>` OR `<type>hybrid</type>`
Then list your search/reason nodes below it.

**EXAMPLES:**

[Example 1 - Type: chain]
User Question: What country was the performer of "Shape of You" born in?
<plan_think>
Step 1 - Entity Analysis: Given = "Shape of You" (Song). Target = Birth country.
Step 2 - Missing Link Analysis: I don't know who performed the song. I must find the performer FIRST, then find that person's birth country. Step 2 cannot start until Step 1 finishes.
Step 3 - Classification: Q1: Does the next step always wait for the previous? YES → `chain`
Step 4 - Node Drafting: Chain the two steps with #ID#. Node 2 uses #1#.
</plan_think>
<plan>
<type>chain</type>
<search id="1">Who performs the song "Shape of You"?</search>
<search id="2">In which country was #1# born?</search>
</plan>

[Example 2 - Type: parallel]
User Question: Who is older, Glenn Hughes or Ross Lynch?
<plan_think>
Step 1 - Entity Analysis: Given = "Glenn Hughes", "Ross Lynch". Target = Who is older (age comparison).
Step 2 - Missing Link Analysis: I need the birth years for both people. These can be searched parallelly. After getting the birth dates, I need a reason node to compare their ages.
Step 3 - Classification: Q1: Does the next step wait? NO. Q2: Can all start simultaneously? YES. Q3: Final answer just comparing results? NO (need reasoning to compare) → `parallel`
Step 4 - Node Drafting: Two parallel search branches → one reason node to compare the ages. #ID# only at the reason node.
</plan_think>
<plan>
<type>parallel</type>
<search id="1">What is the birth date of Glenn Hughes?</search>
<search id="2">What is the birth date of Ross Lynch?</search>
<reason id="3">Compare the birth dates from #1# and #2#. Determine who is older based on their birth years.</reason>
</plan>

[Example 3 - Type: hybrid]
User Question: What movie features both the actor who played Iron Man in 2008 and the actress who played Black Widow?
<plan_think>
Step 1 - Entity Analysis: Given = "Iron Man in 2008", "Black Widow". Target = A movie co-starring both.
Step 2 - Missing Link Analysis: The actor and actress names are unknown. I can search for both simultaneously. But after I get the names, I MUST merge them into ONE new query to find the shared movie. That final search cannot start until the merge step is done.
Step 3 - Classification: Q1: Does the next step wait? NO (first two searches are independent). Q2: Can all start simultaneously? YES. Q3: Final answer just comparing results? NO (need a new search after merge) → `hybrid`
Step 4 - Node Drafting: Two parallel branches → merge step → downstream search. #ID# only at the merge point.
</plan_think>
<plan>
<type>hybrid</type>
<search id="1">Who played the character Iron Man in the 2008 movie?</search>
<search id="2">Who played the character Black Widow in the Marvel Cinematic Universe?</search>
<reason id="3">Person 1 is #1#. Person 2 is #2#. Output the exact names of these two actors so we can search for their co-starred movies.</reason>
<search id="4">What movie features both actors identified in #3# in its cast?</search>
</plan>"""


class PlannerAgent(BaseLLMClient):
    """
    DAG 规划与问题分类 Agent
    
    继承 BaseLLMClient，复用 JADE 的 LLM 调用和 trace 机制。
    功能：
        - 单个问题生成 DAG 执行计划
        - 支持 chain/parallel/hybrid 三种类型
        - 并行处理多个问题
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
        logging.info(f"初始化 PlannerAgent: model={model_name}")

    def _build_messages(self, question: str) -> List[dict]:
        """
        构造生成 DAG 计划的 prompt
        要求输出的 plan 用 <plan>...</plan> 包裹
        """
        return [
            {
                "role": "system",
                "content": PLANNING_SYSTEM_MSG
            },
            {
                "role": "user",
                "content": f"**Question**: {question}"
            }
        ]

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 响应，提取 DAG 节点信息
        
        返回格式：
        {
            "is_valid": bool,
            "plan_type": str,           # chain/parallel/hybrid
            "nodes": [
                {"id": str, "type": str, "query": str},  # type: "search" or "reason"
                ...
            ],
            "raw_output": str,
            "thinking": str
        }
        """
        if not response or not response.strip():
            return {
                "is_valid": False,
                "plan_type": "",
                "nodes": [],
                "raw_output": response,
                "thinking": "",
                "error": "Empty response"
            }

        # 提取 plan 内容
        plan_match = re.search(r'<plan>(.*?)</plan>', response, re.DOTALL | re.IGNORECASE)
        if not plan_match:
            return {
                "is_valid": False,
                "plan_type": "",
                "nodes": [],
                "raw_output": response,
                "thinking": self._extract_thinking(response),
                "error": "No <plan> tag found"
            }

        plan_content = plan_match.group(1).strip()

        # 提取类型
        type_match = re.search(r'<type>(.*?)</type>', plan_content, re.IGNORECASE)
        plan_type = type_match.group(1).strip() if type_match else ""

        # 提取所有节点
        nodes = []
        search_nodes = re.findall(r'<search\s+id="(\d+)">(.*?)</search>', plan_content, re.DOTALL | re.IGNORECASE)
        for node_id, query in search_nodes:
            nodes.append({
                "id": node_id,
                "type": "search",
                "query": query.strip()
            })

        reason_nodes = re.findall(r'<reason\s+id="(\d+)">(.*?)</reason>', plan_content, re.DOTALL | re.IGNORECASE)
        for node_id, query in reason_nodes:
            nodes.append({
                "id": node_id,
                "type": "reason",
                "query": query.strip()
            })

        # 按 id 排序
        nodes.sort(key=lambda x: int(x["id"]))

        # 校验
        is_valid = (
            plan_type in ["chain", "parallel", "hybrid"] and
            len(nodes) > 0 and
            all(node["type"] in ["search", "reason"] for node in nodes)
        )

        return {
            "is_valid": is_valid,
            "plan_type": plan_type,
            "nodes": nodes,
            "raw_output": response,
            "thinking": self._extract_thinking(response)
        }

    def _extract_thinking(self, response: str) -> str:
        """提取 plan_think 思考过程"""
        for tag in ["plan_think", "search_think", "reason_think", "think"]:
            match = re.search(f'<{tag}>(.*?)</{tag}>', response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def get_plan(self, question: str) -> Dict[str, Any]:
        """
        同步获取单个问题的 DAG 计划
        """
        messages = self._build_messages(question)
        response = self.get_response(messages, temperature=self.default_temperature)
        return self.parse_response(response)

    async def get_plan_async(self, question: str) -> Dict[str, Any]:
        """
        异步获取单个问题的 DAG 计划
        """
        messages = self._build_messages(question)
        response = await self.async_get_response(
            messages,
            agent_name="PlannerAgent",
            temperature=self.default_temperature
        )
        return self.parse_response(response)

    async def get_plans_parallel(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        并行获取多个问题的 DAG 计划
        """
        batch_messages = [self._build_messages(q) for q in questions]
        responses = await self.async_batch_get_response(
            batch_messages,
            agent_name="PlannerAgent",
            temperature=self.default_temperature
        )
        return [self.parse_response(resp) for resp in responses]


if __name__ == "__main__":
    import asyncio
    
    agent = PlannerAgent(model_name="gpt-4o")
    
    test_questions = [
        "What country was the performer of 'Shape of You' born in?",
        "Who is older, Glenn Hughes or Ross Lynch?",
        "What movie features both the actor who played Iron Man in 2008 and the actress who played Black Widow?"
    ]
    
    async def test():
        print("=== 测试 PlannerAgent ===\n")
        results = await agent.get_plans_parallel(test_questions)
        for i, result in enumerate(results):
            print(f"问题 {i+1}: {test_questions[i]}")
            print(f"类型: {result['plan_type']}")
            print(f"节点: {result['nodes']}")
            print(f"有效: {result['is_valid']}")
            print("-" * 50)
    
    asyncio.run(test())
