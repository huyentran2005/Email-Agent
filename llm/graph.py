"""LangGraph điều phối luồng hỏi đáp email chỉ đọc."""

from __future__ import annotations

from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core.observability import LangfuseTracer

from .email_assistant import AssistantAnswer, LOCAL_TIMEZONE

if TYPE_CHECKING:
    from .email_assistant import EmailAssistant


class EmailAgentState(TypedDict, total=False):
    """State thô được chia sẻ giữa các node của email agent."""

    question: str
    history: list[dict[str, Any]]
    intent: str
    detail_scope: str
    planned_calls: list[dict[str, Any]]
    previous_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    last_tool_results: list[dict[str, Any]]
    replan_count: int
    max_replans: int
    should_run_tools: bool
    policy_error: str
    answer_message: dict[str, Any]
    audit_log: list[dict[str, Any]]


def _has_useful_result(results: list[dict[str, Any]]) -> bool:
    """Xác định một vòng tool có trả dữ liệu dùng được hay chỉ lỗi/rỗng."""
    for item in results:
        value = item.get("result")
        if isinstance(value, dict) and value.get("error") and len(value) <= 2:
            continue
        if value not in (None, "", [], {}):
            return True
    return False


def _trace_node(name: str, as_type: str = "chain"):
    """Trace một graph node nhưng ẩn nội dung email theo mặc định."""

    def decorator(function):
        @wraps(function)
        def wrapper(self, state: EmailAgentState):
            capture = self.assistant.settings.langfuse_capture_content
            trace_input = state if capture else {
                "question_chars": len(state.get("question", "")),
                "planned_tool_count": len(state.get("planned_calls", [])),
                "tool_result_count": len(state.get("tool_results", [])),
                "replan_count": state.get("replan_count", 0),
            }
            with self.tracer.observation(
                name, as_type=as_type, input=trace_input
            ) as observation:
                result = function(self, state)
                observation.update(
                    output=result
                    if capture
                    else {
                        "updated_state_keys": sorted(result),
                        "planned_tool_count": len(result.get("planned_calls", [])),
                        "tool_result_count": len(
                            result.get("last_tool_results", [])
                        ),
                    }
                )
                return result

        return wrapper

    return decorator


class EmailAgentGraph:
    """Graph plan → policy → tools → observe/replan → answer → audit."""

    def __init__(self, assistant: EmailAssistant):
        """Nhận các dependency sẵn có và biên dịch graph một lần."""
        self.assistant = assistant
        self.tracer = LangfuseTracer(assistant.settings.langfuse_enabled)
        self.graph = self._build()

    def _build(self):
        """Khai báo node, cạnh cố định và hai nhánh điều kiện."""
        workflow = StateGraph(EmailAgentState)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("policy", self._policy)
        workflow.add_node("tools", self._tools)
        workflow.add_node("observe", self._observe)
        workflow.add_node("answer", self._answer)
        workflow.add_node("audit", self._audit)

        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "policy")
        workflow.add_conditional_edges(
            "policy",
            self._route_after_policy,
            {"tools": "tools", "answer": "answer"},
        )
        workflow.add_edge("tools", "observe")
        workflow.add_conditional_edges(
            "observe",
            self._route_after_observe,
            {"policy": "policy", "answer": "answer"},
        )
        workflow.add_edge("answer", "audit")
        workflow.add_edge("audit", END)
        return workflow.compile()

    @_trace_node("email-agent.analyze")
    def _analyze(self, state: EmailAgentState) -> dict[str, Any]:
        """Phân tích câu hỏi và lập kế hoạch read tool đầu tiên."""
        plan = self.assistant.plan_retrieval(
            state["question"], state.get("history"), max_calls=3
        )
        calls = list(plan.tool_calls)
        return {
            "intent": plan.intent,
            "detail_scope": plan.detail_scope,
            "planned_calls": calls,
            "should_run_tools": bool(calls),
            "audit_log": state.get("audit_log", [])
            + [
                {
                    "node": "analyze",
                    "tool_call_count": len(calls),
                    "detail_scope": plan.detail_scope,
                }
            ],
        }

    @_trace_node("email-agent.policy", as_type="guardrail")
    def _policy(self, state: EmailAgentState) -> dict[str, Any]:
        """Chỉ cho phép tool đã khai báo read-only đi qua gateway."""
        allowed = {
            schema["name"]
            for schema in self.assistant.tools.planner_schemas()
            if schema.get("read_only") is True
        }
        calls = state.get("planned_calls", [])
        rejected = [call for call in calls if call.get("name") not in allowed]
        if rejected:
            return {
                "planned_calls": [],
                "should_run_tools": False,
                "policy_error": "Yêu cầu chứa thao tác không thuộc nhóm tool chỉ đọc.",
                "audit_log": state.get("audit_log", [])
                + [{"node": "policy", "decision": "deny"}],
            }
        return {
            "should_run_tools": bool(calls),
            "policy_error": "",
            "audit_log": state.get("audit_log", [])
            + [{"node": "policy", "decision": "allow"}],
        }

    @staticmethod
    def _route_after_policy(state: EmailAgentState) -> str:
        """Bỏ qua tool khi policy từ chối hoặc không có kế hoạch hợp lệ."""
        return "tools" if state.get("should_run_tools") else "answer"

    @_trace_node("email-agent.read-tools", as_type="tool")
    def _tools(self, state: EmailAgentState) -> dict[str, Any]:
        """Thực thi một batch read tool và giữ kết quả trong state."""
        calls = state.get("planned_calls", [])
        results = self.assistant.tools.run_calls(calls)
        return {
            "previous_calls": state.get("previous_calls", []) + calls,
            "tool_results": state.get("tool_results", []) + results,
            "last_tool_results": results,
            "should_run_tools": False,
            "audit_log": state.get("audit_log", [])
            + [{"node": "tools", "executed": [call["name"] for call in calls]}],
        }

    @_trace_node("email-agent.observe")
    def _observe(self, state: EmailAgentState) -> dict[str, Any]:
        """Mở rộng search khi cần; nếu không thì replan lúc lỗi hoặc rỗng."""
        last_results = state.get("last_tool_results", [])
        replan_count = state.get("replan_count", 0)
        max_replans = state.get("max_replans", 0)
        expansion_calls = self.assistant.plan_result_expansion(
            state.get("detail_scope", "none"),
            last_results,
            previous_calls=state.get("previous_calls", []),
            query=state.get("question", ""),
            history=state.get("history"),
        )
        if expansion_calls:
            return {
                "planned_calls": expansion_calls,
                "should_run_tools": True,
                "audit_log": state.get("audit_log", [])
                + [{"node": "observe", "decision": "expand"}],
            }

        if _has_useful_result(last_results) or replan_count >= max_replans:
            return {
                "should_run_tools": False,
                "audit_log": state.get("audit_log", [])
                + [{"node": "observe", "decision": "answer"}],
            }

        calls = self.assistant.replan_tools(
            query=state["question"],
            history=state.get("history"),
            previous_calls=state.get("previous_calls", []),
            tool_results=last_results,
            max_calls=2,
        )
        return {
            "planned_calls": calls,
            "should_run_tools": bool(calls),
            "replan_count": replan_count + 1,
            "audit_log": state.get("audit_log", [])
            + [
                {
                    "node": "observe",
                    "decision": "replan" if calls else "answer",
                }
            ],
        }

    @staticmethod
    def _route_after_observe(state: EmailAgentState) -> str:
        """Đưa mọi kế hoạch thay thế qua policy trước khi chạy lại tool."""
        return "policy" if state.get("should_run_tools") else "answer"

    @_trace_node("email-agent.synthesize")
    def _answer(self, state: EmailAgentState) -> dict[str, Any]:
        """Tạo câu trả lời cuối cùng mà không gọi planner thêm lần nữa."""
        policy_error = state.get("policy_error")
        if policy_error:
            answer = AssistantAnswer.error(policy_error)
        else:
            answer = self.assistant.synthesize_answer(
                state["question"],
                state.get("history"),
                state.get("tool_results", []),
            )
        return {
            "answer_message": answer.to_message(),
            "audit_log": state.get("audit_log", []) + [{"node": "answer"}],
        }

    @_trace_node("email-agent.audit")
    def _audit(self, state: EmailAgentState) -> dict[str, Any]:
        """Ghi audit của lượt chạy vào state; chưa tạo thêm bảng PostgreSQL."""
        tool_rounds = sum(
            1 for item in state.get("audit_log", []) if item.get("node") == "tools"
        )
        return {
            "audit_log": state.get("audit_log", [])
            + [
                {
                    "node": "audit",
                    "finished_at": datetime.now(LOCAL_TIMEZONE).isoformat(),
                    "tool_rounds": tool_rounds,
                }
            ]
        }

    def invoke_state(
        self, query: str, history: list[dict[str, Any]] | None = None
    ) -> EmailAgentState:
        """Chạy graph và trả toàn bộ state để debug hoặc kiểm thử."""
        query = query.strip()
        if not query:
            raise ValueError("Câu hỏi không được để trống")
        initial_state = {
            "question": query,
            "history": history or [],
            "intent": "",
            "detail_scope": "none",
            "planned_calls": [],
            "previous_calls": [],
            "tool_results": [],
            "last_tool_results": [],
            "replan_count": 0,
            "max_replans": min(
                max(0, self.assistant.settings.langgraph_max_replans), 3
            ),
            "should_run_tools": False,
            "policy_error": "",
            "audit_log": [],
        }
        capture = self.assistant.settings.langfuse_capture_content
        trace_input = initial_state if capture else {
            "question_chars": len(query),
            "history_messages": len(history or []),
        }
        with self.tracer.observation(
            "email-agent.question",
            as_type="agent",
            input=trace_input,
            metadata={"read_only": "true"},
        ) as observation:
            state = self.graph.invoke(initial_state)
            observation.update(
                output=state.get("answer_message")
                if capture
                else {
                    "answer_chars": len(
                        str(state.get("answer_message", {}).get("content", ""))
                    ),
                    "source_count": len(
                        state.get("answer_message", {}).get("sources", [])
                    ),
                    "tool_call_count": len(state.get("previous_calls", [])),
                    "replan_count": state.get("replan_count", 0),
                }
            )
            return state

    def invoke(
        self, query: str, history: list[dict[str, Any]] | None = None
    ) -> AssistantAnswer:
        """Chạy graph và chỉ trả đối tượng câu trả lời cho UI hiện tại."""
        state = self.invoke_state(query, history)
        return AssistantAnswer.from_message(state["answer_message"])
