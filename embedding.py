from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
# ✅ 1) 사용할 매뉴얼 PDF 경로들
pdf_paths = [
    "PIC-HyundaiWia_SKT100-200-CNC-Installation-Maintenance.pdf",
    # 필요하면 여기에 다른 CNC 매뉴얼도 추가
    # "CNC_Maintenance_Manual.pdf",
]

docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

print(f"로드한 페이지 수: {len(docs)}")

# ✅ 2) 적당한 크기로 쪼개기
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)
splits = splitter.split_documents(docs)
print(f"청크 개수: {len(splits)}")

# ✅ 3) 임베딩 + 벡터스토어 생성 (한번만)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectordb = Chroma.from_documents(
    splits,
    embedding=embeddings,
    persist_directory="cnc_rag_db",  # 로컬 폴더로 저장
)

print("✅ CNC 매뉴얼 RAG DB 저장 완료: cnc_rag_db/")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory="cnc_rag_db",
    embedding_function=embeddings,
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 4}  # 상위 4개 청크 사용
)