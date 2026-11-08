import argparse
import os
import json
import sys
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import unquote
import yamlium
from dataclasses import dataclass
from collections import defaultdict
from openai import OpenAI

from app.lsp_client import LSPClient

# ==========================================
# 1. 全局状态与常量区
# ==========================================
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
MODEL = "anthropic/claude-haiku-4.5"

# LSP 状态记录
DOC_STATES = {}
_lsp_cursor = 0
_last_shown_diagnostics = {}


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: str
    context: str = ""

    @classmethod
    def from_file(cls, path: str) -> "Skill":
        with open(path, encoding="utf-8") as f:
            meta, body = yamlium.read_markdown(f.read())
            if not meta:
                raise ValueError(f"{path}: meta 为空")
            try:
                meta_dict = dict(meta)
                name = str(meta_dict.get("name", "")).strip()
                desc = str(meta_dict.get("description", "")).strip()
                ctx = str(meta_dict.get("context", "")).strip()
            except Exception as e:
                raise ValueError(f"{path}: 转换失败 {e}")
            if not name or not desc:
                raise ValueError(f"{path}: 缺少 name/description")
            return cls(name, desc, body.lstrip("\n"), os.path.dirname(path), ctx)


# ==========================================
# 2. 提示词与工具定义区
# ==========================================
def build_skills_prompt(root: str = ".claude/skills") -> str:
    skills = []
    if not os.path.isdir(root):
        return "You have no additional skills available."
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        skill_file = os.path.join(folder, "SKILL.md")
        if os.path.isfile(skill_file):
            try:
                skills.append(Skill.from_file(skill_file))
            except Exception:
                continue
    if skills:
        lines = [f"- {s.name}: {s.description}" for s in skills]
        base = "You have access to the following skills:\n" + "\n".join(lines)
        base += "\n\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."
        return base
    return "You have no additional skills available."

def build_user_prompts(raw_input: str) -> list[str]:
    if not raw_input.startswith("/"):
        return [raw_input]
    tokens = raw_input.split()
    skill_names = []
    arg_idx = len(tokens)
    for i, token in enumerate(tokens):
        if token.startswith("/"):
            skill_name = token[1:]
            if os.path.isfile(os.path.join(".claude/skills", skill_name, "SKILL.md")):
                skill_names.append(skill_name)
                continue
        arg_idx = i
        break
    if not skill_names:
        return [raw_input]
    raw_args = " ".join(tokens[arg_idx:])
    args_list = raw_args.split()
    results = []
    for name in skill_names:
        try:
            body = Skill.from_file(os.path.join(".claude/skills", name, "SKILL.md")).body
            env = {"$ARGUMENTS": raw_args}
            for i, arg in enumerate(args_list):
                env[f"${i}"] = env[f"$ARGUMENTS[{i}]"] = arg
            processed_body = re.sub(r'\$(?:ARGUMENTS(?:\[\d+\])?|\d+)', lambda m: env.get(m.group(), ""), body)
            header = f"Skill: {name} (located at .claude/skills/{name})\nPaths in the instructions below are relative to that folder."
            injected = f"{header}\n\n{processed_body}"
            results.append(injected)
        except Exception as e:
            print(f"DEBUG: parse error in {name} -> {e}", file=sys.stderr)
    return results if results else [raw_input]


# ==========================================
# 3. LSP 专属逻辑区
# ==========================================
def sync_lsp_document(lsp_client, abs_path: Path):
    if not lsp_client or not abs_path.exists():
        return
    uri = abs_path.as_uri()
    text = abs_path.read_text(encoding="utf-8")
    ext = abs_path.suffix
    if uri in DOC_STATES:
        new_version = DOC_STATES[uri] + 1
        lsp_client.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": new_version},
            "contentChanges": [{"text": text}]
        })
        DOC_STATES[uri] = new_version
    else:
        lang_id = lsp_client.extension_to_language.get(ext, "python")
        lsp_client.notify("textDocument/didOpen", {
            "textDocument": {"uri": uri, "languageId": lang_id, "version": 1, "text": text}
        })
        DOC_STATES[uri] = 1

def fetch_lsp_diagnostics(lsp_client) -> str:
    global _lsp_cursor, _last_shown_diagnostics
    if not lsp_client:
        return ""
    try:
        res = lsp_client.notifications(_lsp_cursor, 0)
    except Exception:
        return ""
    _lsp_cursor = res.get("cursor", _lsp_cursor)
    latest = {}
    for note in res.get("notifications", []):
        if note.get("method") != "textDocument/publishDiagnostics":
            continue
        params = note.get("params", {})
        uri = params.get("uri")
        if not uri:
            continue
        version = params.get("version")
        if version is not None and version != DOC_STATES.get(uri):
            continue
        latest[uri] = params.get("diagnostics", [])

    report = []
    for uri, diagnostics in latest.items():
        curr = set()
        for d in diagnostics:
            start = d["range"]["start"]
            curr.add((start["line"] + 1, start["character"] + 1, d["message"]))
        new_issues = curr - _last_shown_diagnostics.get(uri, set())
        _last_shown_diagnostics[uri] = curr
        if new_issues:
            res_path = Path(unquote(uri.replace("file://", "")))
            rel_path = os.path.relpath(res_path, Path.cwd())
            lines = [f"{rel_path}:"]
            for l, c, m in sorted(new_issues):
                lines.append(f"  [Line {l}:{c}] {m}")
            report.append("\n".join(lines))

    if not report:
        return ""
    return f"\n\n<new-diagnostics>\nThe following new diagnostic issues were detected:\n\n{'\n\n'.join(report)}\n</new-diagnostics>"


# ==========================================
# 4. 各个工具的执行函数
# ==========================================
def tool_read(args: dict) -> str:
    with open(args["file_path"], "r", encoding="utf-8") as f:
        return f.read()

def tool_write(args: dict, lsp_client) -> str:
    with open(args["file_path"], "w", encoding="utf-8") as f:
        f.write(args["content"])
    result = "OK"
    if lsp_client:
        abs_path = Path(args["file_path"]).resolve()
        sync_lsp_document(lsp_client, abs_path)
        time.sleep(0.5)
        result += fetch_lsp_diagnostics(lsp_client)
    return result

def tool_bash(args: dict) -> str:
    try:
        result = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=30)
        return (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"

def tool_lsp(args: dict, lsp_client) -> str:
    operation = args.get("operation")
    file_path = args.get("filePath")
    line = int(args.get("line", 1))
    character = int(args.get("character", 1))
    
    if not lsp_client:
        return "LSP is unconfigured."
        
    abs_path = Path(file_path).resolve()
    uri = abs_path.as_uri()
    sync_lsp_document(lsp_client, abs_path)
    
    if operation == "goToDefinition":
        res = lsp_client.request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1}
        })
        if not res:
            return "No definition found."
        loc = res[0] if isinstance(res, list) else res
        res_abs_path = Path(unquote(loc["uri"].replace("file://", "")))
        rel_path = os.path.relpath(res_abs_path, Path.cwd())
        start = loc["range"]["start"]
        return f"Defined in {rel_path}:{start['line'] + 1}:{start['character'] + 1}"
        
    elif operation == "findReferences":
        res = lsp_client.request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1},
            "context": {"includeDeclaration": True}
        })
        if not res:
            return "No references found."
        grouped = defaultdict(set)
        for loc in res:
            res_abs_path = Path(unquote(loc["uri"].replace("file://", "")))
            rel_path = os.path.relpath(res_abs_path, Path.cwd())
            start = loc["range"]["start"]
            grouped[rel_path].add((start["line"] + 1, start["character"] + 1))
        
        num_files = len(grouped)
        num_refs = sum(len(locs) for locs in grouped.values())
        lines = [f"Found {num_refs} references across {num_files} files:"]
        for file, locs in grouped.items():
            lines.append(f"\n{file}:")
            for l, c in sorted(locs):
                lines.append(f"  Line {l}:{c}")
        return "\n".join(lines)
    return f"Error: Unknown LSP operation '{operation}'."


# ==========================================
# 5. 主循环
# ==========================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    lsp_client = LSPClient.from_environment()
    if lsp_client:
        print("LSP client connected", file=sys.stderr)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "system", "content": build_skills_prompt()}]
    for prompt in build_user_prompts(args.p):
        messages.append({"role": "user", "content": prompt})

    tools = [
        {"type": "function", "function": {"name": "Read", "description": "Read file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
        {"type": "function", "function": {"name": "Write", "description": "Write file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}},
        {"type": "function", "function": {"name": "Bash", "description": "Run command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "Skill", "description": "Load skill", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "args": {"type": "string"}}, "required": ["name"]}}}
    ]
    if lsp_client:
        tools.append({"type": "function", "function": {"name": "LSP", "description": "Code navigation and analysis", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "enum": ["goToDefinition", "findReferences"]}, "filePath": {"type": "string"}, "line": {"type": "integer"}, "character": {"type": "integer"}}, "required": ["operation", "filePath", "line", "character"]}}})

    while True:
        chat = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        if not chat.choices:
            break

        response = chat.choices[0].message
        msg_dict = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            msg_dict["tool_calls"] = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in response.tool_calls]
        messages.append(msg_dict)

        if not response.tool_calls:
            print(response.content)
            break

        for tc in response.tool_calls:
            try:
                args_dict = json.loads(tc.function.arguments)
                
                if tc.function.name == "Read":
                    result = tool_read(args_dict)
                elif tc.function.name == "Write":
                    result = tool_write(args_dict, lsp_client)
                elif tc.function.name == "Bash":
                    result = tool_bash(args_dict)
                elif tc.function.name == "LSP":
                    result = tool_lsp(args_dict, lsp_client)
                elif tc.function.name == "Skill":
                    skill_name = str(args_dict.get("name", "")).strip()
                    if not skill_name:
                        result = "Error: Missing or empty 'name' parameter."
                    else:
                        try:
                            skill_path = os.path.join(".claude/skills", skill_name, "SKILL.md")
                            if os.path.isfile(skill_path):
                                skill = Skill.from_file(skill_path)
                                
                                # 🔥 Subagent (fork) 核心逻辑
                                if str(skill.context).strip().lower() == "fork":
                                    sub_messages = [{"role": "user", "content": str(skill.body)}]
                                    sub_tools = [t for t in tools if t["function"]["name"] != "Skill"]
                                    
                                    while True:
                                        sub_chat = client.chat.completions.create(
                                            model=MODEL,
                                            messages=sub_messages,
                                            tools=sub_tools,
                                        )
                                        sub_resp = sub_chat.choices[0].message
                                        sub_msg_dict = {"role": "assistant", "content": sub_resp.content or ""}
                                        
                                        if sub_resp.tool_calls:
                                            sub_msg_dict["tool_calls"] = [
                                                {"id": stc.id, "type": stc.type, "function": {"name": stc.function.name, "arguments": stc.function.arguments}}
                                                for stc in sub_resp.tool_calls
                                            ]
                                        sub_messages.append(sub_msg_dict)
                                        
                                        if not sub_resp.tool_calls:
                                            result = sub_resp.content
                                            break
                                            
                                        for stc in sub_resp.tool_calls:
                                            sub_result = ""
                                            try:
                                                sub_args = json.loads(stc.function.arguments)
                                                if stc.function.name == "Read":
                                                    sub_result = tool_read(sub_args)
                                                elif stc.function.name == "Write":
                                                    sub_result = tool_write(sub_args, lsp_client)
                                                elif stc.function.name == "Bash":
                                                    sub_result = tool_bash(sub_args)
                                                elif stc.function.name == "LSP":
                                                    sub_result = tool_lsp(sub_args, lsp_client)
                                                else:
                                                    sub_result = f"Error: Tool {stc.function.name} not supported in subagent."
                                            except Exception as e:
                                                sub_result = f"Error executing tool: {e}"
                                            sub_messages.append({"role": "tool", "tool_call_id": stc.id, "content": sub_result})
                                else:
                                    result = skill.body
                            else:
                                result = f"Error: Skill '{skill_name}' does not exist on disk."
                        except Exception as e:
                            result = f"Error: Failed to load skill '{skill_name}'. Detail: {e}"
                else:
                    result = f"Unknown tool {tc.function.name}"
            except Exception as e:
                result = f"Error: {e}"
            
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

if __name__ == "__main__":
    main()
