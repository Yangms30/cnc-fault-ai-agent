# 🧠 CNC 공정 불량 원인 분석 & 매뉴얼 기반 조치 제안 AI Agent

> CatBoost + SHAP + LLM + RAG + LangGraph + Streamlit 기반  
> **스마트팩토리 CNC 불량 원인 자동 분석 및 현장 조치 가이드 생성 시스템**

---

## ✅ 프로젝트 개요

본 프로젝트는 **CNC 공정 센서 데이터 기반 불량 탐지 → 원인 해석 → 실제 매뉴얼 기반 조치 가이드 자동 생성**까지 수행하는  
**제조 도메인 특화 AI Agent MVP 시스템**입니다.

기존 제조 현장에서는 불량이 발생하면  
- 작업자가 로그를 직접 확인하고
- 경험에 의존해 원인을 추정하며
- 매뉴얼을 수동으로 찾아 조치해야 하는 구조입니다.

본 시스템은 이러한 과정을 **완전 자동화**하여  
> **"불량 탐지 → 원인 추론 → 매뉴얼 기반 조치"를 AI가 일괄 수행**하도록 설계되었습니다.

---

## ✅ 해결하고자 하는 문제

- CNC 공정 중 **불량 발생 시 원인 분석이 전적으로 사람에게 의존**
- 공정 로그 + 센서 데이터 + 매뉴얼 정보가 **분리되어 존재**
- 숙련자 의존도가 높아 **야간/신입/비숙련 상황에서 대응 지연**
- 조치 방법이 **매뉴얼에 존재함에도 실제 현장에서는 즉시 활용되지 못함**

👉 본 프로젝트는 위 문제를 **ML + XAI + LLM + RAG** 구조로 해결합니다.

---

## ✅ 전체 시스템 구조

[센서 데이터 입력]
↓
[CatBoost 불량 예측]
↓
[SHAP 중요 변수 추출]
↓
[LLM Cause Agent]
(불량 유형 + 물리 원인 해석)
↓
[RAG Agent]
(CNC 매뉴얼 PDF 검색)
↓
[Supervisor Agent]
(최종 자동 조치 리포트 생성)
↓
[Streamlit UI 출력]


---

## ✅ 주요 기술 스택

| 구분 | 기술 |
|------|------|
| ML | CatBoost |
| XAI | SHAP |
| LLM | GPT-4o-mini |
| RAG | ChromaDB + OpenAI Embedding |
| Agent Orchestration | LangGraph |
| Backend | LangChain |
| Frontend | Streamlit |
| Data | CNC 공정 센서 데이터 |
| Manual | CNC 설치/유지보수 PDF |

---

## ✅ 프로젝트 폴더 구조

project/
│
├─ agent.py # Cause Agent + RAG Agent + LangGraph
├─ main.py # Streamlit UI
├─ embedding.py # CNC 매뉴얼 PDF → 벡터DB 생성
├─ requirements.txt
├─ catboost_cnc_fault_model.cbm # 학습된 모델
├─ cnc_rag_db/ # Chroma 벡터 DB
├─ sample_cnc_input.csv # 테스트용 샘플 데이터
└─ .env # OpenAI API Key


---

## ✅ 핵심 기능

### 1️⃣ 불량 예측 (ML)
- CNC 센서 데이터를 입력받아 CatBoost 모델로 불량 확률 예측

### 2️⃣ 원인 해석 (XAI)
- SHAP을 통해 불량 예측에 영향을 준 **상위 원인 변수 자동 추출**

### 3️⃣ 원인 유형화 (LLM Cause Agent)
- SHAP 변수 + 센서값 + 공정 물리 규칙을 기반으로  
  **불량 유형 / 물리 현상 / 핵심 원인 자동 추론**

### 4️⃣ 매뉴얼 기반 조치 생성 (RAG Agent)
- CNC 유지보수 매뉴얼 PDF를 벡터화하여  
  **불량 유형에 맞는 실제 조치 절차 자동 검색 & 요약**

### 5️⃣ 최종 자동 리포트 생성 (Supervisor)
- 불량 유형, 원인, 조치 가이드를 종합하여  
  **현장 작업자용 최종 리포트 자동 생성**

---

## ✅ 실행 방법

### 1️⃣ 가상환경 생성 및 활성화

```bash
python3 -m venv venv
source venv/bin/activate
2️⃣ 라이브러리 설치
bash
코드 복사
pip install -r requirements.txt
3️⃣ OpenAI API Key 설정 (.env)
bash
코드 복사
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
4️⃣ CNC 매뉴얼 벡터DB 생성
bash
코드 복사
python embedding.py
5️⃣ Streamlit 실행
bash
코드 복사
streamlit run main.py
✅ Streamlit 데모 화면 구성
CNC 센서 CSV 업로드

불량 확률 출력 (정상 / 불량)

SHAP 상위 원인 변수 시각화

Cause Agent 결과 출력

RAG 기반 매뉴얼 조치 가이드 출력

최종 자동 리포트 출력

✅ 프로젝트 특징 요약
✅ 단순 예측이 아닌 ‘원인 + 조치’까지 자동화

✅ 실제 CNC 유지보수 매뉴얼 기반 RAG 시스템

✅ ML + XAI + LLM + RAG가 분리된 실전형 아키텍처

✅ 현장 적용 가능한 구조

✅ Streamlit 기반 실시간 데모 가능

✅ 향후 확장 계획
다중 공정(FAB, 용접, 프레스) 도메인 확장

다중 불량 Class 분류 모델 확장

실시간 IoT 스트리밍 연동

ERP/MES 연동

예지보전(PHM) 기능 추가

✅ 프로젝트 의의
본 프로젝트는 단순한 AI 모델 시연이 아닌,
제조 현장에서 실제로 문제가 되는 “불량 대응 자동화”를 목표로 설계된 실전형 AI Agent 시스템 MVP입니다.

📌 Author
제조 도메인 특화 LLM-RAG 기반 AI Agent 개발

스마트팩토리 불량 자동 분석 & 조치 시스템

⭐ 사용 기술 한 줄 요약
CatBoost · SHAP · LangChain · LangGraph · ChromaDB · OpenAI · Streamlit
