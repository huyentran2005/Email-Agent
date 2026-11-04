"""Đồng bộ dense+sparse subject index từ PostgreSQL sang Qdrant."""

import argparse

from .subject_embeddings import SubjectEmbeddingIndexer


def main() -> None:
    """Đọc tham số dòng lệnh và đồng bộ subject email sang Qdrant."""
    parser = argparse.ArgumentParser(
        description="Index only email subjects; email bodies are never embedded."
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = SubjectEmbeddingIndexer().index_pending(limit=args.limit)
    print(result)


if __name__ == "__main__":
    main()
