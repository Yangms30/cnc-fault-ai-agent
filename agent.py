import json
from typing import List, Dict, Any, Optional, TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# ===========================
# ✅ 1. 공정 지식 룰북 & 불량명
# ===========================
load_dotenv()
PHYSICAL_RULEBOOK = {
    "X_OutputCurrent": ["절삭 저항 증가", "공구 마모"],
    "Y_OutputCurrent": ["테이블 저항 증가"],
    "S_OutputPower": ["스핀들 부하 상승"],
    "Z_OutputPower": ["Z축 마찰", "볼스크류 간섭"],
    "Z_ActualPosition": ["가공 깊이 오차"],
    "Z_SetAcceleration": ["Z축 급가속 과다"],
    "S_SetVelocity": ["스핀들 속도 과다"],
    "S_DCBusVoltage": ["전원 전압 불안정"],
    "M_sequence_number": ["특정 공정 단계 반복 이상"],
}

FAULT_NAME_MAP = {
    0: "공구 마모 및 절삭 부하 과다형 불량",
    1: "Z축 마찰 및 진동형 불량",
    2: "이송속도-깊이 불일치형 불량",
    3: "전원·전류 이상형 불량",
}

# ===========================
# ✅ 2. LangChain LLM
# ===========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)

# ===========================
# ✅ 3. Cause Agent
# ===========================

class LLMCauseAgent:
    def __init__(self, llm, rulebook, fault_name_map):
        self.llm = llm
        self.rulebook = rulebook
        self.fault_name_map = fault_name_map

    def build_prompt(self, shap_features, sensor_snapshot=None, fault_prob=None):
        feature_list_str = "\n".join(
            [f"- {f['feature']}: SHAP={round(f['value'],4)}" for f in shap_features]
        )

        physical_candidates = {}
        for item in shap_features:
            feat = item["feature"]
            if feat in self.rulebook:
                physical_candidates[feat] = self.rulebook[feat]

        physical_candidates_str = "\n".join(
            [f"- {k}: {', '.join(v)}" for k, v in physical_candidates.items()]
        )

        fault_names_str = "\n".join(self.fault_name_map.values())
        sensor_str = json.dumps(sensor_snapshot, ensure_ascii=False, indent=2)

        return f"""
당신은 CNC 공정 불량을 진단하는 전문가입니다.

[SHAP 중요 변수]
{feature_list_str}

[물리적 현상 후보]
{physical_candidates_str}

[센서 스냅샷]
{sensor_str}

[불량 확률]
{fault_prob}

[가능한 불량 유형]
{fault_names_str}

반드시 JSON 형식으로만 출력하세요:

{{
  "fault_name": "",
  "physical_summary": "",
  "main_causes": ["", ""],
  "confidence": 0.0
}}
"""

    def run(self, shap_features, sensor_snapshot=None, fault_prob=None):
        prompt = self.build_prompt(shap_features, sensor_snapshot, fault_prob)
        result = self.llm.invoke(prompt).content

        try:
            return json.loads(result)
        except:
            return {
                "fault_name": None,
                "physical_summary": result,
                "main_causes": [],
                "confidence": 0.0,
            }


cause_agent = LLMCauseAgent(llm, PHYSICAL_RULEBOOK, FAULT_NAME_MAP)

# ===========================
# ✅ 4. RAG Agent
# ===========================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory="cnc_rag_db",
    embedding_function=embeddings,
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

class CNCRAGAgent:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def run(self, cause_result):
        query = f"""
불량 유형: {cause_result.get("fault_name")}
추정 원인: {', '.join(cause_result.get("main_causes", []))}
현상 요약: {cause_result.get("physical_summary")}
"""

        docs = self.retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
너는 CNC 유지보수 엔지니어다.

[매뉴얼 발췌]
{context}

위 내용을 기반으로 작업자가 바로 실행할 수 있는 '조치 가이드'를 작성하라.
"""

        return self.llm.invoke(prompt).content


rag_agent = CNCRAGAgent(llm, retriever)

# ===========================
# ✅ 5. LangGraph State
# ===========================

class PipelineState(TypedDict, total=False):
    fault_prob: float
    shap_top_features: List[Dict[str, float]]
    sensor_snapshot: Dict[str, float]
    cause_result: Dict[str, Any]
    rag_context: Optional[str]
    final_answer: Optional[str]

# ===========================
# ✅ 6. LangGraph Nodes
# ===========================

def cause_agent_node(state: PipelineState):
    state["cause_result"] = cause_agent.run(
        state["shap_top_features"],
        state["sensor_snapshot"],
        state["fault_prob"],
    )
    return state


def rag_agent_node(state: PipelineState):
    state["rag_context"] = rag_agent.run(state["cause_result"])
    return state


def supervisor_node(state: PipelineState):
    c = state["cause_result"]
    r = state["rag_context"]

    state["final_answer"] = f"""
✅ CNC 불량 자동 분석 리포트

[불량 유형]
{c['fault_name']} (신뢰도: {c['confidence']})

[공정 현상]
{c['physical_summary']}

[추정 원인]
{', '.join(c['main_causes'])}

[매뉴얼 기반 조치 가이드]
{r}
"""
    return state

# ===========================
# ✅ 7. LangGraph Builder
# ===========================

def build_agent_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("CAUSE", cause_agent_node)
    graph.add_node("RAG", rag_agent_node)
    graph.add_node("SUPERVISOR", supervisor_node)

    graph.set_entry_point("CAUSE")
    graph.add_edge("CAUSE", "RAG")
    graph.add_edge("RAG", "SUPERVISOR")
    graph.add_edge("SUPERVISOR", END)

    return graph.compile()