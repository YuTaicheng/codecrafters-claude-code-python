import argparse
import os
import json
import sys
import re
import subprocess
import yamlium
from dataclasses import dataclass
from openai import OpenAI


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: str

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
            except Exception as e:
                raise ValueError(f"{path}: 转换 meta 失败 {e}")
                
            if not name or not desc:
                raise ValueError(f"{path}: 缺少 name/description")
                
            return cls(name, desc, body.lstrip("\n"), os.path.dirname(path))


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


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
    messages = [{"role": "system", "content": skills_ad}]
    
    for prompt in build_user_prompts(args.p):
        messages.append({"role": "user", "content": prompt})

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
        },
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": "Load a skill's instructions into the conversation",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "description": "The name of the skill to use"},
                        "args": {"type": "string", "description": "Optional arguments for the skill"}
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
                elif tc.function.name == "Skill":
                    skill_name = args_dict.get("name")
                    if not skill_name:
                        result = "Error: Missing 'name' parameter."
                    else:
                        try:
                            skill_path = os.path.join(".claude/skills", skill_name, "SKILL.md")
                            skill = Skill.from_file(skill_path)
                            result = skill.body
                        except Exception as e:
                            result = f"Error: Failed to load skill '{skill_name}'. Detail: {e}"
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
