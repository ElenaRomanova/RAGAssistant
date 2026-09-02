class Chunk:
    text: str
    article: str
    chunk_id: str
    tokens_count: int
    doc_id: str
    doc_title: str
    source_url: str

    def __init__(self, text, chunk_id, doc_id, doc_title, source_url, tokens_count=0, article=""):
        self.text = text
        self.chunk_id = chunk_id
        self.tokens_count = tokens_count
        self.article = article
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.source_url = source_url

    def __repr__(self):
        return (f"Chunk(text={self.text!r}, chunk_id={self.chunk_id!r}, tokens_count={self.tokens_count!r}, "
                f"article={self.article!r}, doc_id={self.doc_id!r}, doc_title={self.doc_title!r}, "
                f"source_url={self.source_url!r})")

    def merge(self, other_chunk):
        if not isinstance(other_chunk, Chunk):
            raise ValueError("Can only merge with another Chunk instance.")
        self.text += " " + other_chunk.text
        if self.article == "":
            self.article = other_chunk.article
        else:
            self.article += " " + other_chunk.article
        if self.tokens_count==0:
            self.tokens_count = other_chunk.tokens_count
        else:
            self.tokens_count += other_chunk.tokens_count
