import os
from chromadb import PersistentClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from config import OPENAI_API_KEY, SCRIPT_DIR

# ------------------------------------------------------------------------------
# Chroma 벡터 데이터베이스 설정 및 PDF 텍스트 벡터화
# ------------------------------------------------------------------------------

def setup_chroma_db():
    """
    Chroma 벡터 데이터베이스를 persistent client를 사용해 설정합니다.
    OpenAI 임베딩 함수를 활용하여 PDF 파일 내 텍스트 데이터를 벡터화합니다.
    """
    vdb_path = os.path.join(SCRIPT_DIR, "pdf_vdb")
    chroma_client = PersistentClient(path=vdb_path)
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # EmbeddingFunction 인터페이스 변경에 따른 래퍼 클래스 정의
    class EmbeddingWrapper:
        def __init__(self, model):
            self.model = model

        def __call__(self, input: list) -> list:
            return self.model.embed_documents(input)

    embedding_fn = EmbeddingWrapper(embedding_model)

    pdf_vdb = chroma_client.get_or_create_collection(
        name="pdf",
        embedding_function=embedding_fn
    )
    return pdf_vdb

def add_pdfs_to_db(pdf_vdb, dataset_folder, max_docs=101):
    """
    dataset_folder 내의 모든 PDF 파일을 재귀적으로 검색하여,
    각 PDF의 각 페이지를 별도의 문서로 처리해 벡터 DB에 추가합니다.
    각 문서는 고유 ID를 가지며, 예를 들어 파일명이 "sample.pdf"이고 3페이지라면
    ID는 "sample_page3" 형식으로 생성됩니다.
    """
    ids = []
    documents = []
    metadatas = []
    count = 0

    for root, dirs, files in os.walk(dataset_folder):
        for filename in sorted(files):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, dataset_folder)
                base_pdf_id = os.path.splitext(relative_path.replace(os.sep, "_"))[0]
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                except Exception as e:
                    print(f"PDF 로드 중 에러 발생 ({file_path}):", e)
                    continue
                for i, doc in enumerate(docs):
                    page_text = doc.page_content
                    doc_id = f"{base_pdf_id}_page{i+1}"
                    ids.append(doc_id)
                    documents.append(page_text)
                    metadatas.append({"pdf_id": base_pdf_id, "file_path": file_path, "page": i+1})
                    count += 1
                    if count >= max_docs:
                        break
            if count >= max_docs:
                break
        if count >= max_docs:
            break

    if not ids or not documents:
        print("데이터셋 폴더에서 PDF 파일을 찾지 못했습니다. 폴더 구조와 파일명을 확인해주세요.")
        return

    pdf_vdb.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"총 {count} 개의 페이지가 벡터 DB에 성공적으로 추가되었습니다.")

def ensure_vector_db(pdf_vdb, dataset_folder):
    """
    벡터 DB에 PDF 데이터가 없다면, 로컬 데이터셋 폴더의 PDF 파일들을 추가합니다.
    """
    if pdf_vdb.count() == 0:
        add_pdfs_to_db(pdf_vdb, dataset_folder)
    else:
        print("벡터 DB에 PDF 데이터가 이미 존재합니다. 추가 작업을 생략합니다.")