import argparse
import os
import json
from openai import OpenAI
import subprocess
import yamlium
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    @classmethod
    def from_file(cls, path: str) -> "Skill":
        with open(path, encoding="utf-8") as f:
            meta, body = yamlium.read_markdown(f.read())
            if not meta or not isinstance(meta.get("name"), str) or not isinstance(meta.get("description"), str):
                raise ValueError(f"{path}: 缺少 name/description")
            return cls(meta["name"], meta["description"], body.lstrip("\n"))
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
def build_skills_prompt(root: str = ".claude/skills") -> str:
    skills =[]
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        skill_file = os.path.join(folder, "SKILL.md")
        if os.path.isfile(skill_file):
            try:
                skills.append(Skill.from_file(skill_file))
            except:
                continue
        if skills:
            lines = [f"- {s.name}: {s.description}" for s in skills]
            system = "You have access to the following skills:\n" + "\n".join(lines)
        else:
            system = "You have no additional skills available."
        return system
def bash(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout if result.stdout else result.stderr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = []
    skills_ad = build_skills_prompt()
    messages.append({"role": "system", "content": skills_ad})
    messages.append({"role": "user", "content": args.p})

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read and return the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path to the file to read",
                                }
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
                                "file_path": {
                                    "type": "string",
                                    "description": "The path of the target file to write content to",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content that needs to be written into the target file",
                                }
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
                                "command": {
                                    "type": "string",
                                    "description": "The command to execute"
                                }
                            }
                        }
                    }
                }
            ],
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        response = chat.choices[0].message
        response_message = chat.choices[0].message
        message_dict = {
            "role": response_message.role,
            "content": response_message.content,
        }
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ]
        messages.append(message_dict)
        if not message_dict.get("tool_calls"):
            print(response.content)
            break
        for tc in response.tool_calls:
            args_dict = json.loads(tc.function.arguments)
            if tc.function.name == "Read":
                with open(args_dict["file_path"], "r") as f:
                    result = f.read()
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )

            if tc.function.name == 'Write':
                with open(args_dict["file_path"], "w") as f:
                    try:
                        f.write(args_dict["content"])
                        result = '$2+OK'
                    except Exception as e:
                        result = '-ERROR: {}'.format(e)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            if tc.function.name == 'Bash':
               result = bash(args_dict["command"])
               messages.append(
                   {
                       "role": "tool",
                       "tool_call_id": tc.id,
                       "content": result,
                   }
               )

if __name__ == "__main__":
    main()
