"""
ROSA Agent Manager - 闂備緡鍋勯崐鍧楁晸?ray_trainer.py 闁?RL 闁荤姳鑳堕惌濠囨煙閹帒鍔氶柨?

闁?ROSA 闂佹眹鍔岀€氬箖閸ャ劎鈻?Agent 闂佺懓鍢插Λ妤呭吹椤撱垹绾ч柕澶樺灣缁€澶愭⒑椤愶絽濮冮梺鍙夌矒瀹曪繝寮撮悩宸痪闁荤姴閸熶即寮妶澶嬫櫖閻忕偠鍋愮粻浠嬫煙闂€鎰悽鍋婂鍨緞鐏炲墽鏆犻柣鐘辫兌閻﹪鏌￠崒婊勫殌缂佹唻绻濋弫?- 濠电到缁诲鏁?Agent 闂佸憡濞叉粎鈧凹鍘鹃幃鎯冮悢鍏煎仺?- 濠电到缁诲鏁?Agent 闂佸憡濞叉粎鈧凹鍙冨銊ф崉閸濆嫮鍘?reward 婵烇絽娲犻埀鍟挎繛?
- 闂佸搫鐗冮崑鎾剁磽娴ｅ摜澧㈤柨?final_f1 婵炴潙鍚嬮懝楣冨垂鎼淬劍鐓€鐎广儱娲ㄩ懜鍫曟煙绾版ê浜鹃柨?Agent

Agent 闂佸搫瀚慨鎾儍閻樿绀傜€规洖娲崗鍥煥濞戞澧曢柨?qa_manager/qa.py 闁?ROSA闂佹寧绋戦¨鈧柨?
- "PlanningAgent"              闁?PlannerAgent (DAG 闁荤喐鐟ョ€氭晸?
- "QueryRewriteAgent"          闁?QueryRewriterAgent (闁荤姴娲ㄩ悿鍛存⒑閹绘帞校闁?
- "AnswerGenerationAgent"       闁?SearcherAgent (闂佸搫鍊稿ú锕傚Υ閸屾娑㈠焵椤掑倹?+ 缂備焦绋掑褰掓煙閹増闁?
- "ReasonerAgent"              闁?ReasonerAgent (闂備緡鍋呭Σ鎺旀椤愶箑绠抽柕濞у嫬鈧?
- "AnswerSummarizationAgent"   闁?SynthesizerAgent (闂佸搫鐗冮崑鎾剁磽娴ｅ摜澧㈤柟缍佸畷?
"""

import json
import logging
import re
import httpx
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto
from typing import Dict, Any, List

from qa_manager.ROSA.contract_protocol import (
    build_contract_planner_system_message,
    contract_protocol_metadata,
    parse_contract_plan,
    validate_contract_plan,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_insufficient_conclusion(conclusion: str) -> bool:
    return is_invalid_protocol_conclusion(conclusion)


def remove_trailing_marker(text: str) -> str:
    marker = "<|im_end|>"
    if text.endswith(marker):
        return text[:-len(marker)]
    return text


# Shared values are authoritative for both training and evaluation.
try:
    from qa_manager.ROSA.protocol import (  # noqa: E402, F811
        ANSWER_GENERATION_SYSTEM_MSG,
        FINAL_SYSTEM_MSG,
        QUERY_REWRITE_SYSTEM_MSG,
        REASON_SYSTEM_MSG,
        SEARCH_SYSTEM_MSG,
        PLANNER_SYSTEM_MSG,
        build_answer_generation_messages,
        build_planning_messages,
        build_query_rewrite_messages,
        build_reasoner_messages,
        build_summarization_messages,
        find_reference_ids,
        is_degenerate_text,
        is_invalid_conclusion as is_invalid_protocol_conclusion,
        validate_plan_semantics,
        validate_plan_structure,
        SUBGOAL_FINAL_SYSTEM_MSG,
        build_subgoal_planner_system_message,
        build_subgoal_planning_messages,
        validate_subgoal_plan_semantics,
    )
except ImportError as exc:
    if "find_reference_ids" not in str(exc):
        raise
    from qa_manager.ROSA.protocol import (  # noqa: E402, F811
        ANSWER_GENERATION_SYSTEM_MSG,
        FINAL_SYSTEM_MSG,
        QUERY_REWRITE_SYSTEM_MSG,
        REASON_SYSTEM_MSG,
        SEARCH_SYSTEM_MSG,
        PLANNER_SYSTEM_MSG,
        build_answer_generation_messages,
        build_planning_messages,
        build_query_rewrite_messages,
        build_reasoner_messages,
        build_summarization_messages,
        is_degenerate_text,
        is_invalid_conclusion as is_invalid_protocol_conclusion,
        validate_plan_semantics,
        validate_plan_structure,
    )

    def find_reference_ids(text: str) -> list[str]:
        return re.findall(r"#(\d+)#", str(text))


class ROSA_AgentManager:
    """
    ROSA Agent 缂備胶濯寸槐鏇㈠箖婵犲洤闂紒杈ㄧ箞瀹曟鏁?ray_trainer.py 闂佹眹鍔岀€氭暜閹剧煑?
    Agent 闂佸憡甯楅〃澶愬Υ閸愭櫖闁哥矌閻?ray_trainer.py 婵炴垶鎸婚悾?agent_name 婵炴垶鎸撮崑鎾斥槈閹句壕闁诲海鏁搁幊鎾惰姳閺屻儲鏅鑸电〒缁?
    - "PlanningAgent"              闁?PlannerAgent
    - "QueryRewriteAgent"           闁?QueryRewriterAgent
    - "AnswerGenerationAgent"        闁?SearcherAgent (缂備焦绋掑褰掓煟閵忋垹鏋戦柨?
    - "ReasonerAgent"               闁?ReasonerAgent
    - "AnswerSummarizationAgent"    闁?SynthesizerAgent
    """

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

        # ========== ROSA Agent 闂備焦婢樼粔鍫曟晸?==========
        self.topk = config.get("topk", 3)

        # Prompt 濠碘槅鍨崜婵嬫晸?
        self.Prompts = {
            "PlanningAgent": PLANNER_SYSTEM_MSG,
            "QueryRewriteAgent": QUERY_REWRITE_SYSTEM_MSG,
            "AnswerGenerationAgent": ANSWER_GENERATION_SYSTEM_MSG,
            "ReasonerAgent": REASON_SYSTEM_MSG,
            "AnswerSummarizationAgent": FINAL_SYSTEM_MSG,
        }

        # Planner prompts 闂佸湱椤﹁京鎮锕€鍨傞悗锝庡墮閸嬪秶鈧鍠楀ú鈺冩濞嗘挻鍋ㄩ柕濞垮€楅懝鎯р槈閹惧崬鈧繂螣娴ｇ硶鏋旂痪鏉跨粈?

        # ========== 濠碘槅鍋€閸嬫挾绱掓笟鍨仼濞村吋鍔欏畷?==========
        # 闁诲海鎳撻ˇ宕ｈ箛娑欏殧鐎瑰嫭婢樼徊鍧楁⒑閺夎法肖闁汇倕妫濋弫宥囦沪閻愭闂佸綊鏅茬粈浣虹矈閸ㄩ檮婵°倕鍟板▔銏ゆ晸?(qa_manager.retriever_ips) 闂佸憡绮岄張鏅舵惔銊у祦闁告劖褰冮弬褔鏌ｉ姀銏犳灁缂佸姂瀵?        import json
        def safe_get(cfg, key, default=None):
            # 闁诲繐绻戠换鍡涙儊椤栫偞鍎庨悗娑櫭径宥夋煠閹冩Щ闁?
            val = cfg.get(key, None)
            if val is not None:
                return val
            # 闁诲繐绻戠换鍡涙儊椤栨氨鏆ら悘鐐扮矙閹炽垹鐣￠弶璺ㄦ偧 (cfg.qa_manager.retriever_ips)
            if '.' in key:
                parts = key.split('.')
                obj = cfg
                for p in parts:
                    if hasattr(obj, p):
                        obj = getattr(obj, p)
                    elif isinstance(obj, dict):
                        obj = obj.get(p)
                    else:
                        return default
                return obj if obj is not None else default
            return default

        self.max_nodes_per_turn = int(
            safe_get(
                config,
                "qa_manager.max_nodes_per_turn",
                config.get("max_nodes_per_turn", 5),
            )
        )
        self.planner_prompt_max_nodes = int(
            safe_get(
                config,
                "qa_manager.planner_prompt_max_nodes",
                config.get("planner_prompt_max_nodes", self.max_nodes_per_turn),
            )
        )
        self.contract_preserving_dag_v2 = bool(
            safe_get(
                config,
                "qa_manager.contract_preserving_dag_v2",
                config.get("contract_preserving_dag_v2", False),
            )
        )
        self.model_enable_thinking = safe_get(
            config,
            "data.apply_chat_template_kwargs.enable_thinking",
            None,
        )
        self.plan_format = str(
            safe_get(config, "qa_manager.plan_format", config.get("plan_format", "typed")) or "typed"
        ).strip().lower()
        if self.plan_format not in {"typed", "subgoal"}:
            raise ValueError(f"qa_manager.plan_format must be typed or subgoal, got {self.plan_format!r}")
        if self.plan_format == "subgoal" and self.contract_preserving_dag_v2:
            raise ValueError("qa_manager.plan_format=subgoal is incompatible with contract_preserving_dag_v2")
        if self.plan_format == "subgoal":
            self.Prompts["PlanningAgent"] = build_subgoal_planner_system_message(self.planner_prompt_max_nodes)
            self.Prompts["AnswerSummarizationAgent"] = SUBGOAL_FINAL_SYSTEM_MSG
        elif self.contract_preserving_dag_v2:
            self.Prompts["PlanningAgent"] = build_contract_planner_system_message(
                self.planner_prompt_max_nodes
            )
        else:
            self.Prompts["PlanningAgent"] = build_planning_messages(
                "", max_nodes=self.planner_prompt_max_nodes
            )[0]["content"]

        retriever_urls = safe_get(config, "qa_manager.retriever_urls", config.get("retriever_urls", None))
        max_retrieval_concurrency = int(safe_get(config, "qa_manager.max_retrieval_concurrency", config.get("max_retrieval_concurrency", 16)))
        retrieval_timeout = float(safe_get(config, "qa_manager.retrieval_timeout", config.get("retrieval_timeout", 30.0)))
        ips = safe_get(config, "qa_manager.retriever_ips", config.get("retriever_ips", ['127.0.0.1']))
        retriever_path = safe_get(config, "qa_manager.retriever_path", config.get("retriever_path", "/retrieve"))

        # 婵犮垼娉涚€氭晸?Hydra 闂佸憡绋掗崹婵嬪箮閵堝洦鍋橀悘鐐跺亹閻愬﹪鏌涜箛瀣姢婵炲牊鍨归埀绋掗〃鍫濃槈閹炬潙绲婚柤鍛婃緲椤曪綀绠涢弴鐔烘殸闂佸憡甯楅〃澶愭晸?(闁?'["127.0.0.1"]')
        if retriever_urls is not None:
            if isinstance(retriever_urls, str):
                try:
                    ips_ports = json.loads(retriever_urls)
                except:
                    import ast
                    try:
                        parsed_urls = ast.literal_eval(retriever_urls)
                        ips_ports = list(parsed_urls) if isinstance(parsed_urls, (list, tuple)) else [str(parsed_urls)]
                    except:
                        ips_ports = [retriever_urls]
            else:
                ips_ports = list(retriever_urls)
        elif isinstance(ips, str):
            try:
                ips = json.loads(ips)
            except:
                ips = [ips]

            ips_ports = []
            for ip in ips:
                ips_ports.extend([f"http://{ip}:800{i}{retriever_path}" for i in range(8)])
        else:
            ips_ports = []
            for ip in ips:
                ips_ports.extend([f"http://{ip}:800{i}{retriever_path}" for i in range(8)])
        logging.info(f"[ROSA_AgentManager] 濠碘槅鍋€閸嬫挾绱掓笟鍨伈闁告ǜ鍊楃槐? ips={ips}, retriever_path={retriever_path}, urls={ips_ports}")
        self.retrieval_tool = RetrievalTool(
            api_urls=ips_ports,
            timeout=retrieval_timeout,
            max_concurrency=max_retrieval_concurrency,
        )

        # ========== 闂備焦婢樼粔鍫曟晸?==========
        self.max_prompt_length = safe_get(config, "data.max_prompt_length", config.get("max_prompt_length", 1024))
        self.return_raw_chat = safe_get(config, "data.return_raw_chat", config.get("return_raw_chat", True))
        self.truncation = safe_get(config, "data.truncation", config.get("truncation", "right"))

        logging.info("ROSA_AgentManager initialized")

    # ==================== 闂佸搫绉堕…鍫㈢紦妤ｅ啫绠抽柕澶堝劚缂?1: trans_rawprompt_to_ids ====================

    def trans_rawprompt_to_ids(self, agent_name: str, *args) -> Dict[str, Any]:
        """
        闂佸搫绉烽～澶愭晸?agent_name 闂佸搫鐎氱磽閹炬枼妲堥柛鍠嶇槐鏍ㄩ悙鏉戝姸婵炴潙妫欑粙?token IDs

        闁?ray_trainer.py 闁?async_rollout_single() 闂佺绻掗崢?
        """
        name_mapping = {
            "AnswerGenerationAgent": "SearcherAgent",
        }
        agent_name = name_mapping.get(agent_name, agent_name)

        messages = self._build_messages_by_agent(agent_name, *args)
        return self._messages_to_ids(messages)

    def trans_messages_to_ids(self, messages: List[Dict]) -> Dict[str, Any]:
        """Convert prebuilt chat messages to token IDs for rollout prompts."""
        return self._messages_to_ids(messages)

    def is_insufficient_conclusion(self, conclusion: str) -> bool:
        return is_insufficient_conclusion(conclusion)

    def _build_messages_by_agent(self, agent_name: str, *args) -> List[Dict]:
        """闂佸搫绉烽～澶愭晸?agent_name 闂佸搫鐎氱磽閹惧灈鍋撻悽鍨殌缂併劍鐓￠幆?prompt messages"""

        if agent_name == "PlanningAgent":
            # args: (question: str)
            # 婵炴垶鎸堕崐婵單熸担绯曟灁? 闂佽浜介崕鏌ュ极瑜版帒绀嗛柛鈩冪懆椤妇绱撴担瑙勫鞍闁诲繐閺佸秶浠﹂崜褍闂佸湱椤箓鏌涢妸銉剶闁逞屽墮椤︿即鏁?planner prompt
            question = args[0]

            if getattr(self, "plan_format", "typed") == "subgoal":
                return build_subgoal_planning_messages(question, max_nodes=self.max_nodes_per_turn)

            if self.contract_preserving_dag_v2:
                return [
                    {"role": "system", "content": self.Prompts["PlanningAgent"]},
                    {
                        "role": "user",
                        "content": f"Question: {question}\nGenerate the execution plan.",
                    },
                ]

            return build_planning_messages(
                question, max_nodes=self.max_nodes_per_turn
            )

        elif agent_name == "QueryRewriteAgent":
            # args: (question: str, raw_query: str, step_results: dict)
            # 闁诲酣娼х紞濠囨晸?rosa_server_v2.py 闁?QueryRewriteAgent 闁哄鐗婇幐鎼佸矗閸℃稑鍐€闁绘挸娴风涵鈧?
            question, raw_query, step_results = args[0], args[1], args[2] if len(args) > 2 else {}

            # 濠碘槅鍋€閸嬫捇鏌＄仦璇插姕婵″弶鎮傚畷銉╂晜閼恒儳鐣?#ID# 婵炴挻纰嶇换鍡涙晸?
            return build_query_rewrite_messages(question, raw_query, step_results)

        elif agent_name == "AnswerGenerationAgent" or agent_name == "SearcherAgent":
            # args: (query: str, docs: list)
            query, docs = args
            return build_answer_generation_messages(query, docs)

        elif agent_name == "ReasonerAgent":
            # args: (logic_task: str)
            logic_task = args[0]
            return build_reasoner_messages(logic_task)

        elif agent_name == "AnswerSummarizationAgent":
            # args: (question: str, subqs: list, subas: list)
            question, subqs, subas = args
            return build_summarization_messages(question, subqs, subas)

        raise ValueError(f"闂佸搫鐗忛崣鈧柨?agent_name: {agent_name}")

    def _messages_to_ids(self, messages: List[Dict]) -> Dict[str, Any]:
        """闁?messages 闁哄鍋涙惔濠囨晸?token IDs"""
        template_kwargs = {}
        if self.model_enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(self.model_enable_thinking)
        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        raw_prompt_ids = input_ids[0].tolist()

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        update_dict = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
        }

        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]

        update_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            update_dict["raw_prompt"] = messages

        return update_dict

    # ==================== 闂佸搫绉堕…鍫㈢紦妤ｅ啫绠抽柕澶堝劚缂?2: get_answers_text ====================

    def get_answers_text(self, data: DataProto) -> List[str]:
        """Decode rollout responses from a DataProto."""
        predicted_answers = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences_str = self.tokenizer.decode(valid_response_ids)
            sequences_str = remove_trailing_marker(sequences_str)
            predicted_answers.append(sequences_str)

        return predicted_answers

    # ==================== 闂佸搫绉堕…鍫㈢紦妤ｅ啫绠抽柕澶堝劚缂?3: parse_response ====================

    def parse_response(self, agent_name: str, response_text: str, **kwargs) -> Dict[str, Any]:
        """Parse one agent response into a structured result."""

        name_mapping = {
            "AnswerGenerationAgent": "SearcherAgent",
        }
        agent_name = name_mapping.get(agent_name, agent_name)

        if agent_name == "PlanningAgent":
            return self._parse_planning_response(
                response_text,
                max_nodes=int(kwargs.get("max_nodes", 0) or 0),
                question=str(kwargs.get("question", "") or ""),
            )

        elif agent_name == "QueryRewriteAgent":
            return self._parse_rewrite_response(response_text)

        elif agent_name == "SearcherAgent":
            return self._parse_answer_response(response_text)

        elif agent_name == "ReasonerAgent":
            return self._parse_reason_response(response_text)

        elif agent_name == "AnswerSummarizationAgent":
            return self._parse_synthesis_response(response_text)

        return {"is_valid": False, "error": f"Unknown agent: {agent_name}"}

    def _parse_planning_response(
        self,
        response: str,
        max_nodes: int = 0,
        question: str = "",
    ) -> Dict[str, Any]:
        """Parse a PlanningAgent response."""
        import re

        if getattr(self, "plan_format", "typed") == "subgoal":
            return self._parse_subgoal_planning_response(response, max_nodes=max_nodes, question=question)

        if getattr(self, "contract_preserving_dag_v2", False):
            return self._parse_contract_planning_response(
                response,
                max_nodes=max_nodes,
                question=question,
            )

        # Step 1: 濠碘槅鍋€閸嬫捇鏌＄仦璇插姎闁?<plan> 闂佸搫绉村ú?
        plan_match = re.fullmatch(r'\s*<plan>(.*?)</plan>\s*', response, re.DOTALL | re.IGNORECASE)
        if not plan_match:
            return {"is_valid": False, "workflow": "", "error": "No <plan> tag found", "reason": "missing_plan_tag"}

        if re.search(r'<plan_think>', response, re.IGNORECASE):
            return {"is_valid": False, "workflow": "", "error": "Should not output <plan_think>", "raw_output": response, "reason": "forbidden_plan_think"}

        plan_content = plan_match.group(1).strip()

        # Step 3: 婵炴垶鎸堕崕鏌ユ偋缁嬭娑㈠焵椤掑嫬钃熼柕澶堝劚閺佸爼姊洪鍝勫缂佸姂瀵?- 闂婃搐閹虫﹢濡磋箛娑樺強濠⒀冮叄閻涚暦閸ャ劎鏆?<search> 闁?<reason> 闂佸搫绉村ú?
        # 濠碘槅鍋€閸嬫捇鏌＄仦璇插姕婵″弶鎮傚畷銉╂晝閳ф偤閵娾晛鎹堕柕濞炬櫆婵粍绻涙径瀣闁绘牭绲跨划鐢稿箻閸涢柨?<where>, <electricity>, <output>, <compare> 缂備焦绋戦¨鈧柨?
        illegal_tags = re.findall(r'<(where|electricity|output|compare)(?:\s|>)', plan_content, re.IGNORECASE)
        if illegal_tags:
            return {
                "is_valid": False,
                "workflow": "",
                "error": f"Illegal tags found: {illegal_tags}",
                "raw_output": response,
                "reason": "illegal_nested_structure"
            }

        # Step 4: 濠碘槅鍋€閸嬫捇鏁?<search> 闁?<reason> 闂佸搫瀚崕濠冩叏濠垫挾鎮奸柍銉︼耿楠炲秵鎷呴幖浣瑰皺闂佹寧绋戦悧鍛箔婢舵劖鍤勯柦妯洪妶闈涒攽閸屾侗娈旀繝鈧鍫濈闁哄牏鏁搁柧鍌炴煛瀹ュ懏宸濋梺鍛婂姇閹锋垿鏁?
        # 闂佸湱绮崝鏇°亹閸ョ闁逞屽墴瀵?<search> 闁?<reason> 闂佸搫绉村ú?
        top_level_searches = re.findall(r'<(search)\s+id="(\d+)">(.*?)</\1>', plan_content, re.DOTALL)
        top_level_reasons = re.findall(r'<(reason)\s+id="(\d+)">(.*?)</\1>', plan_content, re.DOTALL)

        if not top_level_searches and not top_level_reasons:
            return {
                "is_valid": False,
                "workflow": "",
                "error": "No valid <search> or <reason> nodes found",
                "raw_output": response,
                "reason": "no_valid_nodes"
            }

        # Step 5: 濠碘槅鍋€閸嬫捇鏌＄仦璇插姤濠碘槅鍙冮幃?ID 闂佸搫瀚崕濠傗槈閹捐泛鍔嬮梺杞扮閻°劑鏁?
        all_nodes = top_level_searches + top_level_reasons
        node_id_nums = sorted(int(node_id) for _, node_id, _ in all_nodes)
        if node_id_nums != list(range(1, len(node_id_nums) + 1)):
            return {
                "is_valid": False,
                "workflow": "",
                "error": f"Node IDs must start at 1 and increase by 1, got: {node_id_nums}",
                "raw_output": response,
                "reason": "non_contiguous_node_ids"
            }
        for node_type, node_id, node_content in all_nodes:
            try:
                id_num = int(node_id)
                if id_num <= 0:
                    return {
                        "is_valid": False,
                        "workflow": "",
                        "error": f"Node ID must be positive integer, got: {node_id}",
                        "raw_output": response,
                        "reason": "invalid_node_id"
                    }
            except ValueError:
                return {
                    "is_valid": False,
                    "workflow": "",
                    "error": f"Node ID must be integer, got: {node_id}",
                    "raw_output": response,
                    "reason": "invalid_node_id"
                }

        for node_type, node_id, node_content in all_nodes:
            inner_illegal = re.findall(r'<(search|reason|where|electricity|compare)(?:\s|>)', node_content, re.IGNORECASE)
            if inner_illegal:
                return {
                    "is_valid": False,
                    "workflow": "",
                    "error": f"Nested <{node_type}> found inside node {node_id}",
                    "raw_output": response,
                    "reason": "nested_nodes_forbidden"
                }

        all_ids = set(node_id for _, node_id, _ in all_nodes)

        for node_type, node_id, node_content in all_nodes:
            refs = find_reference_ids(node_content)
            for ref_id in refs:
                if ref_id not in all_ids:
                    return {
                        "is_valid": False,
                        "workflow": "",
                        "error": f"Reference #{ref_id}# in node {node_id} but node {ref_id} does not exist",
                        "raw_output": response,
                        "reason": "undefined_reference"
                    }
                if int(ref_id) >= int(node_id):
                    return {
                        "is_valid": False,
                        "workflow": "",
                        "error": f"Node {node_id} references non-previous node #{ref_id}#",
                        "raw_output": response,
                        "reason": "future_or_self_reference"
                    }

        # Step 8: 闂佸搫鐎氭晸?nodes 闂佸憡甯楅〃澶愬Υ閸愭櫖閻忕偞鍎抽惁?id 闂佸湱鍎ょ敮鎺楁晸?
        nodes = []

        # 濠电儑缍€椤曆囨晸?search 闂佺厧鎼崐濠氭晸?
        for node_type, node_id, node_content in top_level_searches:
            nodes.append({
                "id": node_id,
                "type": "search",
                "query": node_content.strip()
            })

        # 濠电儑缍€椤曆囨晸?reason 闂佺厧鎼崐濠氭晸?
        for node_type, node_id, node_content in top_level_reasons:
            nodes.append({
                "id": node_id,
                "type": "reason",
                "task": node_content.strip()
            })

        # 闁?id 闂佸湱鍎ょ敮鎺楁晸?
        nodes.sort(key=lambda x: int(x["id"]))

        structure_errors = validate_plan_structure(nodes, max_nodes=max_nodes)
        semantic_errors = validate_plan_semantics(question, nodes) if question else []
        plan_errors = structure_errors + semantic_errors
        if plan_errors:
            return {
                "is_valid": False,
                "workflow": "",
                "error": "; ".join(plan_errors),
                "raw_output": response,
                "nodes": nodes,
                "reason": "invalid_plan",
            }

        return {
            "is_valid": True,
            "raw_output": response,
            "nodes": nodes
        }

    def _parse_subgoal_planning_response(self, response: str, *, max_nodes: int = 0, question: str = "") -> Dict[str, Any]:
        """Parse a Subgoal Plan v1: one <plan> block of <step id="k">sub-question</step> nodes."""

        def invalid(error, reason, nodes=None):
            out = {"is_valid": False, "workflow": "", "error": error, "raw_output": response, "reason": reason}
            if nodes is not None:
                out["nodes"] = nodes
            return out

        plan_match = re.fullmatch(r'\s*<plan>(.*?)</plan>\s*', response, re.DOTALL | re.IGNORECASE)
        if not plan_match:
            return invalid("No <plan> tag found", "missing_plan_tag")
        plan_content = plan_match.group(1)
        steps = re.findall(r'<step\s+id="(\d+)"\s*>(.*?)</step>', plan_content, re.DOTALL | re.IGNORECASE)
        if not steps:
            return invalid("No valid <step> nodes found", "no_valid_nodes")
        # Every tag in the block must belong to a well-formed step: no stray, nested, or typed (<search>/<reason>) nodes.
        if len(re.findall(r'</?[a-zA-Z_]', plan_content)) != 2 * len(steps):
            return invalid("Unexpected tags inside <plan>; only <step id=\"k\"> nodes are allowed", "illegal_tags")
        leftover = re.sub(r'<step\s+id="\d+"\s*>.*?</step>', "", plan_content, flags=re.DOTALL | re.IGNORECASE)
        if leftover.strip():
            return invalid("Text outside <step> nodes inside <plan>", "text_outside_steps")
        ids = [int(step_id) for step_id, _ in steps]
        if ids != list(range(1, len(ids) + 1)):
            return invalid(f"Step IDs must be 1, 2, ... in order, got: {ids}", "non_contiguous_node_ids")
        nodes = []
        for step_id, text in steps:
            for ref_id in find_reference_ids(text):
                if int(ref_id) >= int(step_id):
                    return invalid(f"Step {step_id} references non-previous step #{ref_id}#", "future_or_self_reference")
            nodes.append({"id": step_id, "type": "step", "task": text.strip()})
        errors = validate_plan_structure(nodes, max_nodes=max_nodes)
        errors += validate_subgoal_plan_semantics(question, nodes) if question else []
        if errors:
            return invalid("; ".join(errors), "invalid_plan", nodes)
        return {"is_valid": True, "raw_output": response, "nodes": nodes}

    def _parse_contract_planning_response(
        self,
        response: str,
        *,
        max_nodes: int = 0,
        question: str = "",
    ) -> Dict[str, Any]:
        """Parse and validate the no-declared-out CP-DAGENT plan contract."""
        if not re.fullmatch(
            r"\s*<plan\b[^>]*>.*?</plan>\s*",
            str(response or ""),
            flags=re.DOTALL | re.IGNORECASE,
        ):
            return {
                "is_valid": False,
                "workflow": "",
                "error": "Expected exactly one typed <plan> block",
                "reason": "missing_or_wrapped_contract_plan",
                "raw_output": response,
            }

        contract_plan = parse_contract_plan(response)
        errors = validate_contract_plan(
            question,
            contract_plan,
            max_nodes=max_nodes,
        )
        nodes = []
        for parsed_node in contract_plan.get("nodes", []):
            node = dict(parsed_node)
            if node.get("type") == "search":
                node["query"] = str(node.get("task", ""))
            nodes.append(node)
        metadata = contract_protocol_metadata(contract_plan, errors)
        if errors:
            return {
                "is_valid": False,
                "workflow": "",
                "error": "; ".join(errors),
                "reason": "invalid_contract_plan",
                "raw_output": response,
                "nodes": nodes,
                "contract_plan": contract_plan,
                "contract_protocol": metadata,
            }

        sink_refs = find_reference_ids(
            str(contract_plan.get("attrs", {}).get("sink", ""))
        )
        return {
            "is_valid": True,
            "raw_output": response,
            "nodes": nodes,
            "contract_plan": contract_plan,
            "contract_protocol": metadata,
            "answer_sink_id": sink_refs[0],
            "answer_type": str(contract_plan.get("attrs", {}).get("answer_type", "")),
        }

    def _parse_rewrite_response(self, response: str) -> Dict[str, Any]:
        """Parse a query rewrite response."""
        text = response.strip()
        is_valid = (
            len(text) > 3
            and "\n" not in text
            and "\r" not in text
            and "#" not in text
            and "<" not in text
            and ">" not in text
            and not is_degenerate_text(text)
        )
        return {
            "is_valid": is_valid,
            "query_rewrited": text if is_valid else response
        }

    def _parse_answer_response(self, response: str) -> Dict[str, Any]:
        """Parse an answer generation response."""
        import re

        # 濠碘槅鍋€閸嬫捇鏌＄仦璇插姕婵″弶鎮傚畷銉╂晜閼恒儳鐣崇紓浣稿€烽悞锕傛晸?thinking blocks
        forbidden_think_tags = ['<search_think>', '<think>', '<reasoning>']
        for tag in forbidden_think_tags:
            if tag.lower() in response.lower():
                return {
                    "is_valid": False,
                    "answer": response,
                    "conclusion": response,
                    "error": f"Forbidden thinking block: {tag}",
                    "reason": "forbidden_thinking_block"
                }

        conclusion_match = re.fullmatch(r'\s*<conclusion>(.*?)</conclusion>\s*', response, re.DOTALL | re.IGNORECASE)
        rewrite_match = re.fullmatch(r'\s*<rewrite>(.*?)</rewrite>\s*', response, re.DOTALL | re.IGNORECASE)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
        rewrite_query = rewrite_match.group(1).strip() if rewrite_match else ""

        has_conclusion = bool(conclusion) and not is_insufficient_conclusion(conclusion)
        has_rewrite = bool(rewrite_query) and not is_degenerate_text(rewrite_query)
        is_valid = (has_conclusion or has_rewrite) and not (has_conclusion and has_rewrite)
        return {
            "is_valid": is_valid,
            "answer": conclusion if has_conclusion else response,
            "conclusion": conclusion,
            "rewrite": rewrite_query,
            "has_conclusion": has_conclusion,
            "has_rewrite": has_rewrite,
            "raw_output": response,
        }

    def _parse_reason_response(self, response: str) -> Dict[str, Any]:
        """Parse a reasoner response."""
        import re

        # 濠碘槅鍋€閸嬫捇鏌＄仦璇插姕婵″弶鎮傚畷銉╂晜閼恒儳鐣崇紓浣稿€烽悞锕傛晸?thinking blocks
        forbidden_think_tags = ['<reason_think>', '<think>', '<reasoning>']
        for tag in forbidden_think_tags:
            if tag.lower() in response.lower():
                return {
                    "is_valid": False,
                    "conclusion": response,
                    "answer": response,
                    "error": f"Forbidden thinking block: {tag}",
                    "reason": "forbidden_thinking_block"
                }

        conclusion_match = re.fullmatch(r'\s*<conclusion>(.*?)</conclusion>\s*', response, re.DOTALL | re.IGNORECASE)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

        is_valid = bool(conclusion) and not is_invalid_protocol_conclusion(conclusion)
        return {
            "is_valid": is_valid,
            "conclusion": conclusion if is_valid else response,
            "answer": conclusion if is_valid else response  # 闂佺绻掗崢?answer 闁诲孩绋掗〃?
        }

    def _parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Parse a final synthesis response."""
        import re

        # 濠碘槅鍋€閸嬫捇鏌＄仦璇插姕婵″弶鎮傚畷銉╂晜閼恒儳鐣崇紓浣稿€烽悞锕傛晸?thinking blocks
        forbidden_think_tags = ['<think>', '<reasoning>']
        for tag in forbidden_think_tags:
            if tag.lower() in response.lower():
                return {
                    "is_valid": False,
                    "answer": response,
                    "conclusion": response,
                    "error": f"Forbidden thinking block: {tag}",
                    "reason": "forbidden_thinking_block"
                }

        answer_match = re.fullmatch(r'\s*<answer>(.*?)</answer>\s*', response, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else ""

        is_valid = bool(answer) and not is_invalid_protocol_conclusion(answer)
        return {
            "is_valid": is_valid,
            "answer": answer if is_valid else response,
            "conclusion": answer if is_valid else response  # 闂佺绻掗崢?conclusion 闁诲孩绋掗〃?
        }

    # ==================== 濠碘槅鍋€閸嬫挾绱掓笟鍨仼濞村吋鍔欏畷妤呮煥鐎ｉ柨?====================

    def retrieval(self, query: str, top_k: int = None) -> List[str]:
        """Run retrieval and return formatted documents."""
        if top_k is None:
            top_k = self.topk
        return self.retrieval_tool.query(query, top_k)


class RetrievalTool:
    """Retrieval helper."""
    def __init__(self, api_urls: List[str], timeout: float = 30.0, max_concurrency: int = 16):
        import threading
        self.api_urls = api_urls
        self.timeout = timeout
        max_concurrency = max(1, int(max_concurrency))
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=5.0),
            limits=httpx.Limits(
                max_connections=max_concurrency,
                max_keepalive_connections=max_concurrency,
            ),
        )

    def query(self, query: str, top_k: int = 3) -> List[str]:
        """Call the retrieval API synchronously."""
        last_error = None
        with self._semaphore:
            for url in self.api_urls:
                try:
                    logging.debug(f"[RetrievalTool] URL: {url}, query: {query[:50]}...")
                    resp = self._client.post(
                        url,
                        json={"queries": [query], "topk": top_k, "return_scores": True}
                    )
                    resp.raise_for_status()
                    results = resp.json().get('result', [[]])[0]

                    docs = []
                    for idx, doc in enumerate(results):
                        content = str(doc.get('document', {}).get('contents', ''))
                        title = content.split("\n")[0] if "\n" in content else "Unknown"
                        docs.append(f"[1_{idx+1}]: (Title: {title}) {content}")
                    return docs if docs else [f"[No Documents Found] for '{query}'."]
                except Exception as e:
                    last_error = e
                    continue

        logging.warning(f"Retrieval API error: {last_error}, tried URLs: {self.api_urls}")
        return [f"[Search API Error]: {last_error}"]

