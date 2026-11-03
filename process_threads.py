import argparse
import json

from email_collector.llm_processing import ThreadLLMProcessor


def main() -> None:
    """Khởi tạo cấu hình và thực thi chức năng chính của script."""
    parser = argparse.ArgumentParser(
        description="Extract events, track changes, and summarize email threads via LLM proxy."
    )
    parser.add_argument("--thread-id", type=int, help="Process one internal email_threads.id")
    args = parser.parse_args()

    processor = ThreadLLMProcessor()
    processor.initialize()
    result = (
        processor.process_thread(args.thread_id)
        if args.thread_id is not None
        else processor.process_all()
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
