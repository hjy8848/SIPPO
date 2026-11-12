"""
ROSA DAGExecutor - 依赖图解析与并行调度

纯代码实现，负责：
1. 解析 DAG 节点的依赖关系
2. 将节点分组为可并行执行的批次
3. 调度执行各 Agent (QueryRewriterAgent, SearcherAgent, ReasonerAgent)
"""

import re
import asyncio
import logging
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class DAGNode:
    """DAG 节点"""
    id: str
    node_type: str  # "search" or "reason"
    query: str
    dependencies: Set[str] = field(default_factory=set)  # 依赖的节点 ID 集合


@dataclass
class ExecutionResult:
    """节点执行结果"""
    node_id: str
    node_type: str
    query: str
    resolved_query: str = ""
    search_context: str = ""
    conclusion: str = ""
    raw_output: str = ""
    error: Optional[str] = None


class DAGExecutor:
    """
    DAG 执行器 - 依赖图解析与并行调度
    
    功能：
        - 解析节点的 #ID# 依赖关系
        - 拓扑排序 + 并行分组
        - 调度 QueryRewriterAgent, SearcherAgent, ReasonerAgent
    """
    
    def __init__(
        self,
        query_rewriter_agent,
        searcher_agent,
        reasoner_agent,
        max_nodes_per_turn: int = 5
    ):
        """
        初始化 DAG 执行器
        
        Args:
            query_rewriter_agent: QueryRewriterAgent 实例
            searcher_agent: SearcherAgent 实例
            reasoner_agent: ReasonerAgent 实例
            max_nodes_per_turn: 每个 turn 最大节点数
        """
        self.query_rewriter = query_rewriter_agent
        self.searcher = searcher_agent
        self.reasoner = reasoner_agent
        self.max_nodes_per_turn = max_nodes_per_turn
        logging.info("初始化 DAGExecutor")
    
    def parse_dependencies(self, nodes: List[Dict[str, Any]]) -> List[DAGNode]:
        """
        解析节点的依赖关系
        
        Args:
            nodes: PlannerAgent 返回的节点列表
            
        Returns:
            DAGNode 列表，带依赖信息
        """
        dag_nodes = []
        for node in nodes:
            node_id = node["id"]
            query = node["query"]
            
            # 提取 #ID# 依赖
            dependencies = set(re.findall(r'#(\d+)#', query))
            
            dag_node = DAGNode(
                id=node_id,
                node_type=node["type"],
                query=query,
                dependencies=dependencies
            )
            dag_nodes.append(dag_node)
        
        return dag_nodes
    
    def group_parallel_nodes(self, nodes: List[DAGNode]) -> List[List[DAGNode]]:
        """
        将节点分组为可并行执行的批次
        
        使用拓扑排序，但同一批次的无依赖节点并行执行
        
        Args:
            nodes: DAGNode 列表
            
        Returns:
            分组后的批次列表，每批内的节点可并行执行
        """
        # 复制一份避免修改原数据
        remaining = {node.id: node for node in nodes}
        completed = set()
        batches = []
        
        while remaining:
            # 找出所有依赖都已完成的节点
            ready = [
                node for node_id, node in remaining.items()
                if node.dependencies <= completed
            ]
            
            if not ready:
                # 循环依赖检测
                logging.error(f"检测到循环依赖！剩余节点: {list(remaining.keys())}")
                # 强制添加一个节点以打破循环
                ready = [list(remaining.values())[0]]
            
            # 限制每批节点数
            batch_size = min(len(ready), self.max_nodes_per_turn - sum(len(b) for b in batches))
            batch = ready[:batch_size]
            batches.append(batch)
            
            # 更新已完成集合
            for node in batch:
                completed.add(node.id)
                del remaining[node.id]
        
        return batches
    
    def resolve_variables(
        self,
        query: str,
        step_results: Dict[str, str]
    ) -> str:
        """
        替换查询中的 #ID# 变量
        
        Args:
            query: 带变量的查询
            step_results: 已完成节点的结论 {node_id: conclusion}
            
        Returns:
            变量替换后的查询
        """
        resolved = query
        for var_id, value in step_results.items():
            resolved = resolved.replace(f"#{var_id}#", str(value))
        return resolved
    
    async def execute_node(
        self,
        node: DAGNode,
        resolved_query: str,
        step_results: Dict[str, str],
        original_question: str,
        request_id: str
    ) -> ExecutionResult:
        """
        执行单个节点
        
        Args:
            node: DAG 节点
            resolved_query: 变量替换后的查询
            step_results: 已完成节点的结论
            original_question: 原始问题
            request_id: 请求 ID
            
        Returns:
            ExecutionResult
        """
        result = ExecutionResult(
            node_id=node.id,
            node_type=node.node_type,
            query=node.query,
            resolved_query=resolved_query
        )
        
        try:
            if node.node_type == "search":
                # 语义重构（仅当有依赖时）
                if node.dependencies:
                    rewrite_result = await self.query_rewriter.rewrite_async(
                        original_goal=original_question,
                        draft_query=node.query,
                        resolved_context=step_results
                    )
                    if rewrite_result["is_valid"]:
                        resolved_query = rewrite_result["rewritten_query"]
                        result.resolved_query = resolved_query
                        logging.info(f"[{request_id}] 🔄 语义重构: {resolved_query}")
                
                # 搜索 + 答案提取
                search_result = await self.searcher.search(resolved_query, request_id)
                result.search_context = search_result["search_context"]
                result.conclusion = search_result["conclusion"]
                result.raw_output = search_result["raw_output"]
                
            elif node.node_type == "reason":
                # 逻辑推理
                reason_result = await self.reasoner.reason_async(resolved_query)
                result.conclusion = reason_result["conclusion"]
                result.raw_output = reason_result["raw_output"]
            
            logging.info(f"[{request_id}] ✅ 节点 {node.id} ({node.node_type}): {result.conclusion}")
            
        except Exception as e:
            logging.error(f"[{request_id}] ❌ 节点 {node.id} 执行失败: {e}")
            result.error = str(e)
            result.conclusion = resolved_query  # 保底
        
        return result
    
    async def execute_dag(
        self,
        nodes: List[Dict[str, Any]],
        original_question: str,
        request_id: str = "default"
    ) -> List[ExecutionResult]:
        """
        执行完整的 DAG
        
        Args:
            nodes: PlannerAgent 返回的节点列表
            original_question: 原始问题
            request_id: 请求 ID
            
        Returns:
            所有节点的执行结果列表
        """
        logging.info(f"[{request_id}] 🚀 开始执行 DAG，共 {len(nodes)} 个节点")
        
        # 解析依赖
        dag_nodes = self.parse_dependencies(nodes)
        logging.info(f"[{request_id}] 📊 依赖解析完成: {[n.id for n in dag_nodes]}")
        
        # 并行分组
        batches = self.group_parallel_nodes(dag_nodes)
        logging.info(f"[{request_id}] 📦 分组完成: {len(batches)} 批次")
        for i, batch in enumerate(batches):
            logging.info(f"[{request_id}]   批次 {i+1}: {[n.id for n in batch]}")
        
        # 执行
        all_results = []
        step_results: Dict[str, str] = {}
        
        for batch_idx, batch in enumerate(batches):
            logging.info(f"[{request_id}] ⚡ 执行批次 {batch_idx + 1}/{len(batches)}")
            
            # 解析变量并执行
            tasks = []
            for node in batch:
                resolved_query = self.resolve_variables(node.query, step_results)
                task = self.execute_node(
                    node, resolved_query, step_results,
                    original_question, f"{request_id}-node{node.id}"
                )
                tasks.append(task)
            
            # 并行执行
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in batch_results:
                if isinstance(res, Exception):
                    logging.error(f"[{request_id}] ❌ 任务异常: {res}")
                else:
                    all_results.append(res)
                    step_results[res.node_id] = res.conclusion
        
        # 按节点 ID 排序
        all_results.sort(key=lambda x: int(x.node_id))
        
        logging.info(f"[{request_id}] ✅ DAG 执行完成")
        return all_results


if __name__ == "__main__":
    # 简单测试
    from qa_manager.ROSA.planner_agent import PlannerAgent
    from qa_manager.ROSA.query_rewriter_agent import QueryRewriterAgent
    from qa_manager.ROSA.searcher_agent import SearcherAgent
    from qa_manager.ROSA.reasoner_agent import ReasonerAgent
    
    # 模拟 Agent
    class MockAgent:
        async def rewrite_async(self, **kwargs):
            return {"is_valid": True, "rewritten_query": kwargs.get("draft_query", "")}
        
        async def search(self, query, request_id):
            return {"query": query, "search_context": "", "conclusion": f"Result for: {query}", "raw_output": ""}
        
        async def reason_async(self, query):
            return {"conclusion": f"Reasoned: {query}", "raw_output": ""}
    
    executor = DAGExecutor(
        query_rewriter_agent=MockAgent(),
        searcher_agent=MockAgent(),
        reasoner_agent=MockAgent()
    )
    
    # 测试依赖解析
    test_nodes = [
        {"id": "1", "type": "search", "query": "Who performs Shape of You?"},
        {"id": "2", "type": "search", "query": "In which country was #1# born?"},
        {"id": "3", "type": "reason", "query": "Person is #1#. What country is #2#?"},
        {"id": "4", "type": "search", "query": "Population of #3#"},
    ]
    
    print("=== 测试依赖解析 ===")
    dag_nodes = executor.parse_dependencies(test_nodes)
    for node in dag_nodes:
        print(f"Node {node.id}: deps={node.dependencies}")
    
    print("\n=== 测试并行分组 ===")
    batches = executor.group_parallel_nodes(dag_nodes)
    for i, batch in enumerate(batches):
        print(f"批次 {i+1}: {[n.id for n in batch]}")
