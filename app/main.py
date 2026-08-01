

import argparse
import json
import os
import sys
from typing import List

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
    ChatCompletionFunctionToolParam,
)

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": args.p}

    messages: List[ChatCompletionMessageParam] = [user_message]

    read_tool: ChatCompletionFunctionToolParam = {
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
    }

    tools: List[ChatCompletionToolUnionParam] = [read_tool]

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=messages,
        tools=tools,
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    if chat.choices[0].message.content:
        print(chat.choices[0].message.content)

    for tc in chat.choices[0].message.tool_calls or []:
        args = json.loads(tc.function.arguments)
        if tc.function.name == "Read":
            with open(args["file_path"]) as f:
                print(f.read())


if __name__ == "__main__":
    main()