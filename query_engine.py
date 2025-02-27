# ------------------------------------------------------------------------------
# 사용자 JSON 데이터를 이용한 체인 실행 및 쿼리 생성
# ------------------------------------------------------------------------------

def query_db(pdf_vdb, query_text, n_results=2):
    """
    벡터 DB에 대해 주어진 쿼리 텍스트를 사용하여 유사 PDF 문서를 검색합니다.
    결과에는 metadatas, documents, distances가 포함됩니다.
    """
    results = pdf_vdb.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )
    return results