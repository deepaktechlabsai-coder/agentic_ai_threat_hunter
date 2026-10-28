from __future__ import annotations

import ipaddress
import json
import operator
import os
import re
from datetime import datetime
from typing import Annotated, Any, Dict, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# Import tools
from tools.web_tools import web_search, visit_page
from tools.atomic_red_team import AtomicRedTeamTool
from tools.sigma_tool import SigmaTool
from tools.virustotal_tool import check_virustotal as _vt_check


# =====================================================================
# ZERO TRUST POLICY GATE
# Every tool call is checked BEFORE it runs:
#   1. Least privilege: each agent may use only its own tools
#   2. Risk scoring: dangerous commands raise the score
#   3. Decision: allow or deny (score >= DENY_THRESHOLD is denied)
#   4. Audit log: every decision is written to audit_log.jsonl
# =====================================================================

AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl")
DENY_THRESHOLD = int(os.getenv("POLICY_DENY_THRESHOLD", "80"))

# Least privilege: which tools each agent is allowed to use
AGENT_PERMISSIONS: Dict[str, set] = {
    "ThreatIntel": {"web_search", "visit_page", "check_virustotal"},
    "Hunter": {"execute_splunk_search", "run_velociraptor_query"},
    "RedTeam": {"atomic_red_team"},
    "Detection": {"sigma_rule"},
    "Responder": {"block_ip", "isolate_host", "disable_user"},
}

# Base risk of each tool (0-100)
TOOL_BASE_RISK: Dict[str, int] = {
    "web_search": 5,
    "visit_page": 10,
    "check_virustotal": 5,
    "sigma_rule": 5,
    "execute_splunk_search": 15,
    "run_velociraptor_query": 20,
    "atomic_red_team": 50,
    "block_ip": 60,
    "isolate_host": 70,
    "disable_user": 70,
}

# Dangerous SPL commands (they change or export data, or run code)
DANGEROUS_SPL = re.compile(
    r"\|\s*(delete|outputlookup|outputcsv|collect|sendemail|script|run|map|dbxquery|sendalert)\b",
    re.IGNORECASE,
)
# Dangerous VQL functions (they run commands or change the endpoint)
DANGEROUS_VQL = re.compile(
    r"\b(execve|shell|powershell|cmd|hunt_add|collect_client|rm|kill|upload)\s*\(",
    re.IGNORECASE,
)

PROTECTED_USERS = {"administrator", "root", "system", "admin", "localsystem"}


def _audit(entry: Dict[str, Any]) -> None:
    """Write one decision to the audit log. Never crash the agent if logging fails."""
    try:
        entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        print(f"Audit log write failed: {e}")


def policy_check(agent: str, tool_name: str, payload: str) -> Dict[str, Any]:
    """Zero trust decision for one tool call. Returns {decision, risk, reason}."""
    payload = payload or ""
    risk = TOOL_BASE_RISK.get(tool_name, 40)
    reasons = []

    # 1) Least privilege
    allowed = AGENT_PERMISSIONS.get(agent, set())
    if tool_name not in allowed:
        result = {"decision": "deny", "risk": 100,
                  "reason": f"{agent} is not allowed to use {tool_name} (least privilege)"}
        _audit({"agent": agent, "tool": tool_name, "payload": payload[:300], **result})
        return result

    # 2) Risk scoring
    if tool_name == "execute_splunk_search" and DANGEROUS_SPL.search(payload):
        risk += 70
        reasons.append("dangerous SPL command (delete/outputlookup/collect/script...)")
    if tool_name == "run_velociraptor_query" and DANGEROUS_VQL.search(payload):
        risk += 70
        reasons.append("dangerous VQL function (execve/shell/powershell...)")

    # 3) Decision
    decision = "deny" if risk >= DENY_THRESHOLD else "allow"
    result = {"decision": decision, "risk": min(risk, 100),
              "reason": "; ".join(reasons) if reasons else "within policy"}
    _audit({"agent": agent, "tool": tool_name, "payload": payload[:300], **result})
    return result


def _gated(agent: str, tool_name: str, fn, payload: str):
    """Run fn(payload) only if the policy gate allows it."""
    verdict = policy_check(agent, tool_name, payload)
    if verdict["decision"] != "allow":
        return f"POLICY DENIED (risk {verdict['risk']}): {verdict['reason']}. Do not retry this query. Use a safe read-only query instead."
    return fn(payload)


# =====================================================================
# State
# =====================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    instructions: str
    tool_output: Optional[Dict[str, Any]]


MAX_SUPERVISOR_STEPS = 10   # stops endless loops in hunt mode
MAX_ADAPTIVE_RETRIES = 3    # how many times Hunter may retry a failed/empty query


# =====================================================================
# Supervisor
# =====================================================================

class SupervisorOutput(BaseModel):
    thought: str = Field(description="The reasoning behind the decision")
    next_agent: Literal["ThreatIntel", "Hunter", "RedTeam", "Detection", "Responder", "FINISH"] = Field(
        description="The next agent to act, or FINISH if the user's request is satisfied"
    )
    instructions_for_agent: str = Field(
        description="Specific instructions for the selected agent. If FINISH, provide the final answer."
    )


VALID_AGENTS = ["ThreatIntel", "Hunter", "RedTeam", "Detection", "Responder", "FINISH"]


def build_supervisor_node(model_name: str):
    system_prompt = """You are the Supervisor (Manager) of a cybersecurity threat hunting and response team.
Your goal is to orchestrate the following specialized agents:

1. **ThreatIntel**: Research external threats, CVEs, actor TTPs, and IOC reputation (VirusTotal).
2. **Hunter**: Search internal logs (Splunk) and endpoint state (Velociraptor).
3. **RedTeam**: Execute Atomic Red Team tests to simulate attacks. If the user mentions "simulate", "atomic", "T1003", or "attack", YOU MUST CHOOSE THIS AGENT.
4. **Detection**: Search and manage Sigma detection rules.
5. **Responder**: Take response actions (block_ip, isolate_host, disable_user). These need human approval. Use ONLY when the user asks for a response action, or when a threat is clearly confirmed by evidence.

**Rules**:
- You do not execute tools yourself. You delegate to agents.
- You can chain multiple agents. E.g., ThreatIntel -> Hunter -> Responder.
- When you have enough information to answer the user, or if the task is complete, choose "FINISH".
- Provide clear, concise instructions to the agents.
- If the user asks a general question, use ThreatIntel or answer directly if you know (then FINISH).

**Routing Logic**:
- 'simulate', 'test', 'attack', or a T-code (like T1003) -> RedTeam
- 'search logs', 'find events', 'investigate' -> Hunter
- 'research', 'google', 'check virustotal', 'is this ip malicious' -> ThreatIntel
- 'sigma rules', 'detection coverage' -> Detection
- 'block ip', 'isolate host', 'disable user', 'contain', 'respond' -> Responder

**Hunt mode**:
If the user gives a high-level goal like "hunt for APT29" or "hunt for PowerShell abuse", plan several steps on your own:
  ThreatIntel (learn the threat's TTPs and IOCs) -> Detection (find related Sigma rules) -> Hunter (search Splunk/Velociraptor for those IOCs) -> FINISH with a clear conclusion.
Do NOT choose Responder in hunt mode unless the user asked for a response. Only recommend it in your final answer.

**Output Format**:
You MUST output a valid JSON object matching this schema:
{{
  "thought": "Your reasoning here...",
  "next_agent": "AgentName" (or "FINISH"),
  "instructions_for_agent": "Instructions..."
}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Return ONLY the JSON object."),
    ])

    llm = ChatOllama(model=model_name, format="json")

    async def supervisor_node(state: AgentState) -> AgentState:
        messages = state["messages"]

        # Safety cap: stop endless supervisor loops
        steps = sum(1 for m in messages if isinstance(m, AIMessage) and str(m.content).startswith("**"))
        if steps >= MAX_SUPERVISOR_STEPS:
            print("WARNING: Supervisor step limit reached. Finishing.")
            return {"next_agent": "FINISH", "instructions": "Step limit reached."}

        response = await llm.ainvoke(prompt.format_messages(messages=messages))
        content = response.content
        print(f"DEBUG: Supervisor Raw Output: {content}")

        try:
            data = json.loads(content)
            next_agent = data.get("next_agent", "FINISH")
            instructions = data.get("instructions_for_agent", "")

            if next_agent not in VALID_AGENTS:
                print(f"WARNING: Supervisor returned invalid agent '{next_agent}'. Defaulting to FINISH.")
                next_agent = "FINISH"
        except Exception as e:
            print(f"Supervisor JSON Parse Error: {e}. Content: {content}")
            next_agent = "FINISH"
            instructions = "Error parsing supervisor decision. Terminating."

        return {"next_agent": next_agent, "instructions": instructions}

    return supervisor_node


# =====================================================================
# Worker node wrapper
# =====================================================================

def create_agent_node(agent_name: str, agent_graph):
    """Wraps a worker graph as a node in the supervisor graph."""
    async def agent_node(state: AgentState) -> AgentState:
        instructions = state.get("instructions", "")
        messages = list(state["messages"])

        if instructions:
            messages.append(HumanMessage(content=instructions))

        result = await agent_graph.ainvoke({"messages": messages})
        new_messages = result["messages"][len(messages):]

        if not new_messages:
            return {
                "messages": [AIMessage(content=f"**{agent_name}**: No response.")],
                "next_agent": "Supervisor",
            }

        last_message = new_messages[-1]
        final_message = AIMessage(content=f"**{agent_name}**: {last_message.content}")
        return {
            "messages": new_messages[:-1] + [final_message],
            "next_agent": "Supervisor",
        }

    return agent_node


# =====================================================================
# Threat Intel agent (web + VirusTotal)
# =====================================================================

def _detect_ioc_type(ioc: str) -> str:
    ioc = ioc.strip()
    try:
        ipaddress.ip_address(ioc)
        return "ip"
    except ValueError:
        pass
    if re.fullmatch(r"[A-Fa-f0-9]{32}|[A-Fa-f0-9]{40}|[A-Fa-f0-9]{64}", ioc):
        return "hash"
    return "url"


def build_intel_agent(model_name: str):
    @tool("check_virustotal")
    def check_virustotal_ioc(ioc: str) -> str:
        """Check an IP address, file hash (MD5/SHA1/SHA256) or URL on VirusTotal. Input is the IOC string only."""
        verdict = policy_check("ThreatIntel", "check_virustotal", ioc)
        if verdict["decision"] != "allow":
            return f"POLICY DENIED: {verdict['reason']}"
        api_key = os.getenv("VT_API_KEY")
        if not api_key:
            return "Error: VT_API_KEY is not set"
        try:
            result = _vt_check(ioc.strip(), ioc_type=_detect_ioc_type(ioc), api_key=api_key)
            return json.dumps(result, default=str)[:4000]
        except Exception as e:
            return f"Error calling VirusTotal: {e}"

    tools = [web_search, visit_page, check_virustotal_ioc]

    system_prompt = """You are a Threat Intelligence Researcher.
Your goal is to find information about CVEs, Threat Actors, and IOCs.
Use the available tools: web_search, visit_page, and check_virustotal (for IP, hash, URL reputation).
IMPORTANT: After gathering information, you MUST summarize your findings in a clear, concise manner.
Do not just dump raw data. Explain what it means for the security posture.
If researching a threat actor, list their main techniques (TTPs) and indicators so the Hunter can search for them.
"""
    return create_react_agent(ChatOllama(model=model_name), tools, prompt=system_prompt)


# =====================================================================
# Red Team agent (human approval before it runs)
# =====================================================================

def build_red_team_agent(model_name: str):
    tools = [AtomicRedTeamTool()]

    async def red_team_node(state: AgentState):
        messages = state["messages"]

        system_prompt_text = """You are a Red Team Operator.
Your goal is to execute Atomic Red Team tests to simulate attacks.
You have access to a tool called 'atomic_red_team'.
YOU MUST USE THIS TOOL to execute tests. DO NOT just describe what you would do.

1. First, identify the correct test (Technique ID) based on the user's request.
2. Call the 'atomic_red_team' tool with action='execute_test' and the technique_id.
   - Example: {"action": "execute_test", "technique_id": "T1003"}
   - YOU MUST PROVIDE 'action'. Valid actions: 'execute_test', 'list_tests'.
3. WAIT for the tool output.
4. Summarize the execution results IN DETAIL.
   - List EACH test command that was executed.
   - State whether it PASSED or FAILED.
   - Provide the return code and relevant output for each.
   - Do NOT just say "Total tests executed". Show the breakdown.
"""
        filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        final_messages = [SystemMessage(content=system_prompt_text)] + filtered_messages

        llm = ChatOllama(model=model_name).bind_tools(tools)
        response = await llm.ainvoke(final_messages)
        return {"messages": [response]}

    def _fix_action(args: dict) -> dict:
        """If action is missing/invalid but technique_id exists, assume execute_test."""
        action = args.get("action")
        if (not action or str(action).strip() == "None" or action not in ["execute_test", "list_tests"]) \
                and "technique_id" in args:
            args["action"] = "execute_test"
        return args

    async def tool_node(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        tool_obj = AtomicRedTeamTool()

        # Standard tool call
        if last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            args = tool_call["args"]

            verdict = policy_check("RedTeam", "atomic_red_team", json.dumps(args, default=str))
            if verdict["decision"] != "allow":
                return {"messages": [ToolMessage(tool_call_id=tool_call["id"],
                                                 content=f"POLICY DENIED: {verdict['reason']}")]}

            if "query" in args and isinstance(args["query"], str):
                query_str = args["query"]
                try:
                    query_json = json.loads(query_str)
                    if isinstance(query_json, dict):
                        query_str = json.dumps(_fix_action(query_json))
                except Exception:
                    pass
                result = await run_in_threadpool(tool_obj.invoke, query_str)
            else:
                action = args.get("query") or args.get("action")
                if action in ["execute_test", "list_tests"] and not args.get("action"):
                    args["action"] = action
                args = _fix_action(args)
                result = await run_in_threadpool(tool_obj.invoke, json.dumps(args))

            return {"messages": [ToolMessage(tool_call_id=tool_call["id"], content=str(result))]}

        # Model wrote the tool call as JSON text instead of a real tool call
        content = last_message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        if content.startswith("{") and ("atomic_red_team" in content or "technique_id" in content):
            try:
                try:
                    args = json.loads(content)
                    if isinstance(args, dict):
                        if "arguments" in args and isinstance(args["arguments"], dict):
                            args = args["arguments"]
                        content = json.dumps(_fix_action(args))
                except Exception:
                    pass

                verdict = policy_check("RedTeam", "atomic_red_team", content)
                if verdict["decision"] != "allow":
                    return {"messages": [HumanMessage(content=f"Tool Execution Result: POLICY DENIED: {verdict['reason']}")]}

                result = await run_in_threadpool(tool_obj.invoke, content)
                return {"messages": [HumanMessage(content=f"Tool Execution Result: {result}")]}
            except Exception as e:
                print(f"Failed to parse hallucinated tool call: {e}")

        return {"messages": []}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"

        content = last_message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        if content.startswith("{") and ("atomic_red_team" in content or "technique_id" in content):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", red_team_node)
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    return workflow.compile()


# =====================================================================
# Detection agent
# =====================================================================

def build_detection_agent(model_name: str):
    tools = [SigmaTool()]
    system_prompt = """You are a Detection Engineer.
Your goal is to find or create detection rules (Sigma, SPL, VQL).
1. Search for relevant Sigma rules.
2. Convert them to SPL or VQL if requested.
3. Provide the rule content or query to the user.
"""
    return create_react_agent(ChatOllama(model=model_name), tools, prompt=system_prompt)


# =====================================================================
# Responder agent (ORCHESTRATION: response actions, simulated)
# Runs only after human approval (interrupt_before in the main graph).
# Actions are SIMULATED: they are written to response_actions.jsonl.
# =====================================================================

RESPONSE_LOG_PATH = os.getenv("RESPONSE_LOG_PATH", "response_actions.jsonl")


def _record_response(action: str, target: str, reason: str) -> str:
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "action": action,
        "target": target,
        "reason": reason,
        "mode": "SIMULATED",
        "status": "executed",
    }
    try:
        with open(RESPONSE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Response log write failed: {e}")
    return f"[SIMULATED] {action} executed on {target}. Reason: {reason}. Logged to {RESPONSE_LOG_PATH}."


def build_responder_agent(model_name: str):
    @tool("block_ip")
    def block_ip(ip: str, reason: str = "Malicious IP") -> str:
        """Block an IP address at the firewall (simulated). Provide the IP and the reason."""
        verdict = policy_check("Responder", "block_ip", ip)
        if verdict["decision"] != "allow":
            return f"POLICY DENIED: {verdict['reason']}"
        try:
            addr = ipaddress.ip_address(ip.strip())
        except ValueError:
            return f"POLICY DENIED: '{ip}' is not a valid IP address."
        if addr.is_loopback or addr.is_private or addr.is_multicast or addr.is_unspecified:
            _audit({"agent": "Responder", "tool": "block_ip", "payload": ip, "decision": "deny",
                    "risk": 100, "reason": "protected address range"})
            return f"POLICY DENIED: {ip} is a private/loopback address and is protected from blocking."
        return _record_response("block_ip", str(addr), reason)

    @tool("isolate_host")
    def isolate_host(hostname: str, reason: str = "Suspected compromise") -> str:
        """Isolate a host from the network (simulated). Provide the hostname and the reason."""
        verdict = policy_check("Responder", "isolate_host", hostname)
        if verdict["decision"] != "allow":
            return f"POLICY DENIED: {verdict['reason']}"
        if not hostname.strip():
            return "POLICY DENIED: hostname is empty."
        return _record_response("isolate_host", hostname.strip(), reason)

    @tool("disable_user")
    def disable_user(username: str, reason: str = "Suspected account compromise") -> str:
        """Disable a user account (simulated). Provide the username and the reason."""
        verdict = policy_check("Responder", "disable_user", username)
        if verdict["decision"] != "allow":
            return f"POLICY DENIED: {verdict['reason']}"
        if username.strip().lower() in PROTECTED_USERS:
            _audit({"agent": "Responder", "tool": "disable_user", "payload": username, "decision": "deny",
                    "risk": 100, "reason": "protected account"})
            return f"POLICY DENIED: '{username}' is a protected account and cannot be disabled."
        return _record_response("disable_user", username.strip(), reason)

    tools = [block_ip, isolate_host, disable_user]

    system_prompt = """You are a Security Response Engineer (SOAR).
You take containment actions using these tools: block_ip, isolate_host, disable_user.
Rules:
- Act ONLY on targets given in the instructions. Never invent an IP, host, or user.
- Call exactly the tool that matches the request.
- If a tool returns POLICY DENIED, do NOT try to bypass it. Explain the denial to the user.
- After acting, summarize what was done, on which target, and why.
"""
    return create_react_agent(ChatOllama(model=model_name), tools, prompt=system_prompt)


# =====================================================================
# Hunter agent (zero trust gate + adaptive retry)
# =====================================================================

def _is_failed_result(result: Any) -> bool:
    """True if a tool result is an error or empty, so the Hunter should try again."""
    text = str(result).strip()
    if not text or text in ("[]", "{}", "null", "None"):
        return True
    low = text.lower()
    if low.startswith("error") or '"error"' in low[:80]:
        return True
    return False


def _retry_count_this_task(messages) -> int:
    """Count [ADAPTIVE] hints since the last instruction given to this agent."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and not str(m.content).startswith("Tool Execution Result"):
            break
        if "[ADAPTIVE]" in str(getattr(m, "content", "")):
            count += 1
    return count


def build_hunter_agent(model_name: str, splunk_fn, velociraptor_fn):
    """Hunter agent: zero trust gated tools + adaptive retry on errors/empty results."""

    class SplunkTool(BaseModel):
        query: str = Field(description="The SPL query to execute")

    class VeloTool(BaseModel):
        query: str = Field(description="The VQL query to execute")

    @tool("execute_splunk_search", args_schema=SplunkTool)
    def execute_splunk_search(query: str):
        """Execute a Splunk SPL query to find historical logs."""
        return _gated("Hunter", "execute_splunk_search", splunk_fn, query)

    @tool("run_velociraptor_query", args_schema=VeloTool)
    def run_velociraptor_query(query: str):
        """Execute a Velociraptor VQL query for live endpoint forensics."""
        return _gated("Hunter", "run_velociraptor_query", velociraptor_fn, query)

    tools = [execute_splunk_search, run_velociraptor_query]

    def _run(name: str, args: dict) -> str:
        q = (args or {}).get("query")
        if not q:
            return "Error: Missing query"
        if name == "execute_splunk_search":
            return str(_gated("Hunter", "execute_splunk_search", splunk_fn, q))
        if name == "run_velociraptor_query":
            return str(_gated("Hunter", "run_velociraptor_query", velociraptor_fn, q))
        return "Error: Unknown tool"

    def _with_adaptive_hint(result: str, messages) -> str:
        """If the result failed or is empty, tell the model to fix the query and retry."""
        if str(result).startswith("POLICY DENIED"):
            return result
        if _is_failed_result(result):
            tries = _retry_count_this_task(messages)
            if tries < MAX_ADAPTIVE_RETRIES:
                return (f"{result}\n[ADAPTIVE] The query failed or returned no data "
                        f"(attempt {tries + 1} of {MAX_ADAPTIVE_RETRIES}). "
                        f"Fix or broaden the query (check syntax, index, time range) and try again.")
            return f"{result}\n[ADAPTIVE] Retry limit reached. Stop and report what you found."
        return result

    async def hunter_node(state: AgentState):
        llm = ChatOllama(model=model_name).bind_tools(tools)
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def tool_node(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        # Standard tool calls
        if last_message.tool_calls:
            results = []
            for tool_call in last_message.tool_calls:
                raw = await run_in_threadpool(_run, tool_call["name"], tool_call["args"])
                content = _with_adaptive_hint(raw, messages)
                results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(content)))
            return {"messages": results}

        # Model wrote the tool call as JSON text instead of a real tool call
        content = last_message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        if content.startswith("{") and ("execute_splunk_search" in content or "run_velociraptor_query" in content):
            try:
                data = json.loads(content)
                raw = await run_in_threadpool(_run, data.get("name"), data.get("arguments", {}))
                result = _with_adaptive_hint(raw, messages)
                return {"messages": [HumanMessage(content=f"Tool Execution Result: {result}")]}
            except Exception as e:
                print(f"Failed to parse hallucinated tool call: {e}")

        return {"messages": []}

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        # Stop if retry limit is reached
        if _retry_count_this_task(messages) > MAX_ADAPTIVE_RETRIES:
            return END

        if last_message.tool_calls:
            return "tools"

        content = last_message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        if content.startswith("{") and ("execute_splunk_search" in content or "run_velociraptor_query" in content):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", hunter_node)
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    return workflow.compile()


# =====================================================================
# Main graph
# =====================================================================

def build_hierarchical_graph(
    splunk_execute_fn: Any,
    velociraptor_fn: Any,
    supervisor_model: str,
    worker_model: str,
):
    intel_agent = build_intel_agent(worker_model)
    red_team_agent = build_red_team_agent(worker_model)
    detection_agent = build_detection_agent(worker_model)
    responder_agent = build_responder_agent(worker_model)
    hunter_agent = build_hunter_agent(worker_model, splunk_execute_fn, velociraptor_fn)

    supervisor_node = build_supervisor_node(supervisor_model)

    graph = StateGraph(AgentState)

    graph.add_node("Supervisor", supervisor_node)
    graph.add_node("ThreatIntel", create_agent_node("ThreatIntel", intel_agent))
    graph.add_node("Hunter", create_agent_node("Hunter", hunter_agent))
    graph.add_node("RedTeam", create_agent_node("RedTeam", red_team_agent))
    graph.add_node("Detection", create_agent_node("Detection", detection_agent))
    graph.add_node("Responder", create_agent_node("Responder", responder_agent))

    graph.add_edge(START, "Supervisor")

    def route_supervisor(state: AgentState):
        next_node = state["next_agent"]
        if next_node == "FINISH":
            return END
        return next_node

    graph.add_conditional_edges(
        "Supervisor",
        route_supervisor,
        {
            "ThreatIntel": "ThreatIntel",
            "Hunter": "Hunter",
            "RedTeam": "RedTeam",
            "Detection": "Detection",
            "Responder": "Responder",
            END: END,
        },
    )

    # Workers always return to the Supervisor
    for worker in ["ThreatIntel", "Hunter", "RedTeam", "Detection", "Responder"]:
        graph.add_edge(worker, "Supervisor")

    # Human-in-the-loop: pause BEFORE risky agents so the user can approve or deny.
    checkpointer = MemorySaver()
    return graph.compile(interrupt_before=["RedTeam", "Responder"], checkpointer=checkpointer)