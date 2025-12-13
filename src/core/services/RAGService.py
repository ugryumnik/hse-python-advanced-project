class RAGService:
    def __init__(self):
        pass

    def query(self, question: str) -> tuple[str, list[dict]]:
        answer = f"Answer to: {question}"
        sources = [{"filename": "example.pdf", "page": 1}]
        return answer, sources
