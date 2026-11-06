import argparse
import os
import json
import sys
from openai import OpenAI
import subprocess
import yamlium
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str

    @classmethod
    def from_file(cls, path: str) -> "Skill":
        with open(path, encoding="utf-8") as f:
            content = f.read()
            

        meta, body = yamlium.read_markdown(content)
        
        if not meta:
            raise ValueError("yamlium 返回的 meta 为空")
            
        # 强制使用 dict() 和 str()，防止 yamlium 返回自定义的 Object
        try:
            meta_dict = dict(meta)
            name = str(meta_dict.get("name", "")).strip()
            desc = str(meta_dict.get("description", "")).strip()
        except Exception as e:
            raise ValueError(f"强制转换 meta 失败: {e}")
            
        if not name or not desc:
            raise ValueError(f"缺少 name 或 description (实际解析: {meta})")
            
        return cls(name, desc, body.lstrip("\n"))


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def build_skills_prompt(root: str = ".claude/skills") -> str:
    skills = []
    if not os.path.isdir(root):
        print(f"DEBUG: {root} does not exist!", file=sys.stderr)
        return "You have no additional skills available."
        
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        skill_file = os.path.join(folder, "SKILL.md")
        if os.path.isfile(skill_file):
            try:
                skills.append(Skill.from_file(skill_file))
            except Exception as e:
                # 把跳过的原因打印到 stderr，方便我们排查
                print(f"⚠️ Skipping invalid skill {skill_file}: {e}", file=sys.stderr)
                continue
                
    if skills:
        lines = [f"- {s.name}: {s.description}" for s in skills]
        system = "You have access to the following skills:\n" + "\n".join(lines)
    else:
        system = "You have no additional skills available."
        
    # 打印最终生成的 system prompt 到 stderr 确认
    print(f"DEBUG System Prompt:\n{system}", file=sys.stderr)
    return system

def build_user_prompt(raw_input:str):
    if not raw_input.startswith("/"):
        return raw_input
    parts = raw_input.split(" ", 1)
    skill_name = parts[0][1:]  # 去掉 '/'
    raw_args = parts[1].strip() if len(parts) > 1 else ""
    args_list = raw_args.split() if raw_args else ""
    skill_path = os.path.join(".claude/skills", skill_name, "SKILL.md")
    try:
        skill = Skill.from_file(skill_path)
        injected = skill.body
        env = {"$ARGUMENTS": raw_args}
        for i, arg in enumerate(args_list):
            env[f"${i}"] = env[f"$ARGUMENTS[{i}]"] = arg
        return re.sub(r'\$(?:ARGUMENTS(?:\[\d+\])?|\d+)', lambda m: env.get(m.group(), ""), injected)')
    except Exception as e:
        print(f"DEBUG: parse error {e}", file=sys.stderr)
        return raw_input
def bash(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    skills_ad = build_skills_prompt()
    messages = [
        {"role": "system", "content": skills_ad},
        {"role": "user", "content": build_user_prompt(args.p)}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "The path to the file to read"}
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a specified file, will overwrite existing content if the file already exists",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "The path of the target file to write content to"},
                        "content": {"type": "string", "description": "The content that needs to be written into the target file"}
                    },
                    "required": ["file_path", "content"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {"type": "string", "description": "The command to execute"}
                    }
                }
            }
        }
    ]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools,
        )

        if not chat.choices:
            raise RuntimeError("no choices in response")

        response = chat.choices[0].message
        
        message_dict = {
            "role": response.role,
            "content": response.content or "",
        }
        
        if response.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
            
        messages.append(message_dict)
        
        if not response.tool_calls:
            print(response.content)
            break
            
        for tc in response.tool_calls:
            result = ""
            try:
                args_dict = json.loads(tc.function.arguments)
                
                if tc.function.name == "Read":
                    with open(args_dict["file_path"], "r", encoding="utf-8") as f:
                        result = f.read()
                elif tc.function.name == "Write":
                    with open(args_dict["file_path"], "w", encoding="utf-8") as f:
                        f.write(args_dict["content"])
                    result = "OK"
                elif tc.function.name == "Bash":
                    result = bash(args_dict["command"])
                else:
                    result = f"Error: Unknown tool {tc.function.name}"
            except Exception as e:
                result = f"Error executing tool: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

if __name__ == "__main__":
    main()
