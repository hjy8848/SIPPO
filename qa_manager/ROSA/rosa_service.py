"""
ROSA 服务入口 - 完整问答流程示例

展示如何组合使用 5 个 Agent + DAGExecutor 实现完整的多跳问答。
"""

import asyncio
import logging
from typing import Dict, Any, List

from qa_manager.ROSA.planner_agent import PlannerAgent
from qa_manager.ROSA.query_rewriter_agent import QueryRewriterAgent
from qa_manager.ROSA.searcher_agent import SearcherAgent
from qa_manager.ROSA.reasoner_agent import ReasonerAgent
from qa_manager.ROSA.synthesizer_agent import SynthesizerAgent
from qa_manager.ROSA.dag_executor import DAGExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ROSAService:
    """
    ROSA 问答服务 - 完整流程
    
    流程：
    1. PlannerAgent → 生成 DAG 计划
    2. DAGExecutor → 解析依赖 + 并行调度执行
    3. SynthesizerAgent → 最终答案综合
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        retriever_url: str = "http://127.0.0.1:8000/retrieve"
    ):
        # 初始化 5 个 Agent
        self.planner = PlannerAgent(model_name=model_name)
        self.query_rewriter = QueryRewriterAgent(model_name=model_name)
        self.searcher = SearcherAgent(model_name=model_name, retriever_url=retriever_url)
        self.reasoner = ReasonerAgent(model_name=model_name)
        self.synthesizer = SynthesizerAgent(model_name=model_name)
        
        # 初始化 DAG 执行器
        self.dag_executor = DAGExecutor(
            query_rewriter_agent=self.query_rewriter,
            searcher_agent=self.searcher,
            reasoner_agent=self.reasoner
        )
        
        logging.info("ROSA Service 初始化完成")
    
    async def ask(self, question: str, request_id: str = "default") -> Dict[str, Any]:
        """
        处理单个问题
        
        Args:
            question: 用户问题
            request_id: 请求 ID
            
        Returns:
            {
                "request_id": str,
                "question": str,
                "plan": {...},           # PlannerAgent 结果
                "node_results": [...],   # DAGExecutor 执行结果
                "final_answer": str,
                "is_valid": bool
            }
        """
        logging.info(f"[{request_id}] 🚀 开始处理: {question}")
        
        # ========== Step 1: Planning ==========
        logging.info(f"[{request_id}] 🧠 Phase 1: Planning...")
        plan_result = await self.planner.get_plan_async(question)
        
        if not plan_result["is_valid"]:
            logging.error(f"[{request_id}] ❌ 规划失败")
            return {
                "request_id": request_id,
                "question": question,
                "plan": plan_result,
                "node_results": [],
                "final_answer": "",
                "is_valid": False,
                "error": "Planning failed"
            }
        
        logging.info(f"[{request_id}] 📋 计划类型: {plan_result['plan_type']}")
        logging.info(f"[{request_id}] 📋 节点数: {len(plan_result['nodes'])}")
        
        # ========== Step 2: DAG Execution ==========
        logging.info(f"[{request_id}] ⚡ Phase 2: DAG Execution...")
        node_results = await self.dag_executor.execute_dag(
            nodes=plan_result["nodes"],
            original_question=question,
            request_id=request_id
        )
        
        # ========== Step 3: Synthesis ==========
        logging.info(f"[{request_id}] 📝 Phase 3: Final Synthesis...")
        
        # 转换为 SynthesizerAgent 需要的格式
        synthesis_input = [
            {
                "id": r.node_id,
                "type": r.node_type,
                "query": r.query,
                "conclusion": r.conclusion
            }
            for r in node_results
        ]
        
        synthesis_result = await self.synthesizer.synthesize(
            original_question=question,
            node_results=synthesis_input,
            request_id=request_id
        )
        
        logging.info(f"[{request_id}] 🎯 最终答案: {synthesis_result['answer']}")
        
        return {
            "request_id": request_id,
            "question": question,
            "plan": {
                "type": plan_result["plan_type"],
                "nodes": plan_result["nodes"]
            },
            "node_results": [
                {
                    "id": r.node_id,
                    "type": r.node_type,
                    "query": r.query,
                    "resolved_query": r.resolved_query,
                    "search_context": r.search_context,
                    "conclusion": r.conclusion,
                    "error": r.error
                }
                for r in node_results
            ],
            "final_answer": synthesis_result["answer"],
            "is_valid": synthesis_result["is_valid"]
        }
    
    async def ask_parallel(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        并行处理多个问题
        """
        tasks = [
            self.ask(question, f"req-{i}")
            for i, question in enumerate(questions)
        ]
        return await asyncio.gather(*tasks)


async def demo():
    """演示用例"""
    
    # 初始化服务
    service = ROSAService(model_name="gpt-4o")
    
    # 测试问题
    test_questions = [
        # Chain 类型
        "What country was the performer of 'Shape of You' born in?",
        # Parallel 类型
        "Who is older, Glenn Hughes or Ross Lynch?",
        # Hybrid 类型
        "What movie features both the actor who played Iron Man in 2008 and the actress who played Black Widow?"
    ]
    
    print("=" * 60)
    print("ROSA 问答系统演示")
    print("=" * 60)
    
    for i, question in enumerate(test_questions):
        print(f"\n【问题 {i+1}】{question}")
        print("-" * 60)
        
        result = await service.ask(question, f"demo-{i}")
        
        print(f"计划类型: {result['plan']['type']}")
        print(f"节点数: {len(result['node_results'])}")
        for node in result["node_results"]:
            print(f"  [{node['type']}] {node['query'][:50]}... → {node['conclusion'][:30]}...")
        print(f"最终答案: {result['final_answer']}")
        print()


if __name__ == "__main__":
    asyncio.run(demo())
