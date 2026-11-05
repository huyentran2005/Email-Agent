"""Giao diện chat cho trợ lý đọc và tổng hợp email."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from zoneinfo import ZoneInfo

from llm.email_assistant import AssistantAnswer, EmailAssistant
from llm.llm_proxy import LLMProxyError


SUGGESTIONS = {
    ":blue[:material/summarize:] Tổng hợp email mới": (
        "Tổng hợp các email mới nhất và cho tôi biết việc gì cần chú ý."
    ),
    ":green[:material/task_alt:] Việc cần làm": (
        "Liệt kê các yêu cầu hoặc công việc đang mở trong email."
    ),
    ":orange[:material/schedule:] Deadline": (
        "Có deadline hoặc cuộc hẹn nào cần lưu ý không?"
    ),
}
LOCAL_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
PIPELINE_IMAGE = Path(__file__).resolve().parent / "assets" / "email_agent_pipeline.svg"


st.set_page_config(
    page_title="Trợ lý email",
    page_icon=":material/mark_email_read:",
    layout="centered",
)


@st.cache_resource
def get_assistant() -> EmailAssistant:
    """Khởi tạo một trợ lý dùng chung cho tiến trình Streamlit."""
    return EmailAssistant()


def clear_conversation() -> None:
    """Xóa lịch sử hội thoại của tab hiện tại."""
    st.session_state["messages"] = []


def render_sources(answer: AssistantAnswer) -> None:
    """Hiển thị các email đã được dùng làm nguồn cho câu trả lời."""
    if not answer.sources:
        return
    source_box = st.expander(
        f"{len(answer.sources)} email nguồn",
        icon=":material/mail:",
        type="compact",
    )
    with source_box:
        for source in answer.sources:
            received = source.received_at.astimezone(LOCAL_TIMEZONE).strftime(
                "%d/%m/%Y %H:%M"
            )
            st.markdown(
                f"**{source.subject}**  \n"
                f"{source.from_address} · {received}"
            )


def render_attachments(answer: AssistantAnswer, message_key: str) -> None:
    """Hiển thị nút tải lazy cho các attachment đã vượt qua security policy."""
    if not answer.attachments:
        return
    assistant = get_assistant()
    st.caption("Tệp đính kèm an toàn")
    for attachment in answer.attachments:
        size_kb = max(1, round(attachment.size / 1024))
        st.download_button(
            label=attachment.filename,
            data=lambda item=attachment, service=assistant: service.download_attachment(
                item
            ),
            file_name=attachment.filename,
            mime=attachment.mime_type,
            key=f"attachment-{message_key}-{attachment.attachment_id}",
            help=f"{size_kb:,} KB",
            on_click="ignore",
            icon=":material/download:",
            width="content",
        )


st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Trợ lý email")
    st.caption(
        "Câu trả lời được tạo từ email và kết quả extract đã lưu trong PostgreSQL."
    )
    st.button(
        "Cuộc trò chuyện mới",
        icon=":material/add_comment:",
        on_click=clear_conversation,
        width="stretch",
    )

st.title("Trợ lý email")
st.caption("Đọc, lọc và tổng hợp thông tin từ email bằng hội thoại tự nhiên.")

with st.expander(
    "Pipeline xử lý",
    icon=":material/account_tree:",
    type="compact",
):
    st.image(
        PIPELINE_IMAGE,
        width="stretch",
        alt=(
            "Pipeline Analyze, Policy, Tools, Observe, Answer và Audit; "
            "Observe có thể quay lại Policy để mở rộng hoặc lập kế hoạch lại."
        ),
    )
    st.caption(
        "Nhánh lặp chỉ chạy khi cần mở rộng dữ liệu hoặc lập kế hoạch lại; "
        "mọi tool đều đi qua Policy trước khi thực thi."
    )

if not st.session_state["messages"]:
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        st.markdown(
            "Chào bạn! Mình có thể tổng hợp email mới, tìm việc cần làm, "
            "deadline hoặc tra cứu thông tin."
        )

for message_index, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            previous_answer = AssistantAnswer.from_message(message)
            st.markdown(previous_answer.content)
            render_attachments(previous_answer, f"history-{message_index}")
            render_sources(previous_answer)
        else:
            st.markdown(message["content"])

prompt = st.chat_input(
    "Hỏi về email của bạn...",
    submit_mode="disable",
    key="email_chat_input",
)

if not st.session_state["messages"]:
    suggestions = st.empty()
    selected_suggestion = suggestions.pills(
        "Câu hỏi gợi ý",
        list(SUGGESTIONS),
        label_visibility="collapsed",
    )
    if selected_suggestion:
        prompt = SUGGESTIONS[selected_suggestion]
    if prompt:
        suggestions.empty()

if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state["messages"].append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=":material/robot_2:"):
        try:
            with st.status(":shimmer[Đang đọc email]", type="compact") as status:
                assistant = get_assistant()
                answer = assistant.answer(
                    prompt,
                    history=st.session_state["messages"][-7:-1],
                )
                status.update(
                    label=f"Đã đối chiếu {len(answer.sources)} email",
                    state="complete",
                )
            st.markdown(answer.content)
            render_attachments(answer, f"current-{len(st.session_state['messages'])}")
            render_sources(answer)
        except (SQLAlchemyError, LLMProxyError, ValueError) as exc:
            answer = AssistantAnswer.error(
                "Mình chưa thể đọc email lúc này. Hãy kiểm tra kết nối PostgreSQL "
                "và cấu hình LLM trong `.env`."
            )
            st.error(answer.content, title="Không thể trả lời")
            with st.expander("Chi tiết kỹ thuật", icon=":material/diagnosis:"):
                st.code(str(exc), language=None)

    st.session_state["messages"].append(answer.to_message())
