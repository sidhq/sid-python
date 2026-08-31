def bm25_snippet_with_stride(
    query: str,
    content: str,
    window_size: int = 50,
    stride: int = 10,
    language: str = "english",
) -> tuple[int, int]: ...
