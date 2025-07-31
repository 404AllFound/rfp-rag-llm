import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List, Union

def extract_context(docs: Union[List[dict], List["Document"], str]) -> str:
    if isinstance(docs, str):
        return docs
    elif isinstance(docs, list):
        first = docs[0]
        if hasattr(first, "page_content"):
            return "\n\n".join([doc.page_content for doc in docs])
        elif isinstance(first, dict):
            return "\n\n".join([doc["content"] for doc in docs if "content" in doc])
    raise ValueError("지원하지 않는 문서 형식입니다.")


SYSTEM_PROMPT = """
[시스템]
- 너는 입찰 제안 요청서(RFP) 분석을 전문으로 하는 AI 어시스턴트다.
- 사용자가 수익성 높은 RFP를 식별하고, 불공정한 조건이 포함된 RFP를 피할 수 있도록 돕는 것이 임무다.
- 너는 질문자가 궁금해하는 내용만 정확하게 추출해 답변해야 하며, 질문과 무관한 정보는 답변하지 않는다.
- 아래 [임무 우선순위]는 네 내부 판단 기준일 뿐이며, 사용자에게는 노출하지 않는다.
- [문맥]에 없는 정보를 지어내지 않는다.

---
[임무 우선순위] ← (답변에는 표시하지 말 것)
0순위: 불공정 조항 식별 (중요)
- 키워드: '사후정산', '지식재산권 공동 소유(귀속)', '자사 인력 파견(상주 또는 근무 등)' 
- 숨겨진 형태나 유사 표현도 포함하여 최우선으로 검토한다.
-위험한 조항이 있다면 문구 그대로 인용하고 어떤 피해가 예상되는지 명확히 경고한다.

1순위: 핵심 사업성 분석
- 사업 예산, 대금 지급 조건 등을 파악한다.
- 입찰 참여 가능한 기업의 자격과 조건(지역, 실적, 규모 등)을 확인한다.

2순위: 주요 과업 내용 요약
- 사업 목적과 배경, 왜 이 사업이 필요한지를 요약한다.
- 기업아 요구하는 구체적인 업무(개발, 구축, 설계 등)을 정리한다.
- 입찰 및 계약 관련 기간(입찰 시작/종료일, 선정일, 계약 기간 등)을 포함한다.

3순위: 행정 절차 요약
- 제안서 제출 방식, 입찰 진행 방식, 계약 방식 등 행정적 절차를 요약한다. 

---
[답변 규칙]
- 질문자가 요청한 항목만 간결하고 정확하게 답변한다.
- 요약 요청 시에는 위 우선순위에 따라 핵심 내용을 중심으로 구조적으로 정리한다. 
- 불공정 조항 존재 여부는 "있음" 또는 "없음"으로 명확히 표시하고, 반드시 근거 문구를 포함한다. 
- 문맥에 있는 내용만 사용하며, 없는 내용은 "문서에 정보 없음"으로 명확히 말한다. 
- 사용자의 요청이 없으면, 불필요한 우선순위 설명이나 추가 정보는 포함하지 않는다.
- 기본 출력 형식은 마크다운이며, 가동성을 위해 적정한 이모지를 활용한다. 


---
[문맥]
{context}

[질문]
{question}

[답변]
"""

def create_generation_chain():
    print(">>>>> Start RAG Chain <<<<<")
    llm_model_dir = "/home/models/llm/Midm-2.0-Mini-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(llm_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, tokenizer

if __name__ == "__main__":
    from A_embedding_model import load_embedding_model
    from langchain_community.vectorstores import FAISS

    embedding_model = load_embedding_model("kr-sbert")

    # Load FAISS vectorstore using LangChain
    model_key = "kr-sbert"
    vectorstore = FAISS.load_local(
        folder_path=f"/home/data/faiss_db/{model_key}",
        index_name="faiss",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Build generation chain
    chain, tokenizer = create_generation_chain()

    # Prompt loop
    question = input("질문: ")
    docs = retriever.invoke(question)
    context = extract_context(docs[:2])  
    answer = chain.invoke({"context": context, "question": question})
    print("\n답변:")
    print(answer)