import json
import re

import requests


def unicode_to_symbols(text):
    def replace_unicode(match):
        return chr(int(match.group(1), 16))

    # 使用正则表达式查找并替换Unicode转义序列
    return re.sub(r'\\u([0-9a-fA-F]{4})', replace_unicode, text)


def extract_content(json_data):
    # Parse the JSON-like data
    data = json.loads(json_data)

    # Extract the content field from the 'choices' array
    content = data['choices'][0]['message']['content']

    return content


def fetch_code_suggestions(input_code):
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": f"""
          Next, you will play the role of a programming expert and give various suggestions for fixing the code based on the defects marked in the code. First check the code segment, $ represents the area around the code defect, <code></code> represents the code segment, and <text></text> represents the text segment that marks the cause of the code error. Give various suggestions for fixing the problem based on the input code, programming language, and prompts. Also use <code></code> to mark the code segment, and <text></text> to mark all text explanations except the code. Only reply to the code, and nothing else
    Q: Programming language: c++/n<code>if $a>5$ {{$print("big")$;}}></code> \n <text>The if statement in C++ needs to put the condition in parentheses () and the statement block in curly braces {{}}. In addition, the print function in C++ should be std::cout. </text>
    A:<text> 1. Repair suggestion 1: </text>\n<code> if (a>5) {{cout<<"big";}} </code>
    2.<text>Repair suggestion 2: <text>\n<code> if (a>5) {{}}cout<<"big";</code>

    Q: {input_code}
    A:
                """
            }
        ],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer sk-facfuhlcjbqmwxeepypsdoovbdfeqlwshnzktbseklafqifk"
    }

    response = requests.post(url, json=payload, headers=headers)

    output_text = unicode_to_symbols(extract_content(response.text))
    # print(unicode_to_symbols(response.text))
    print(output_text)
    print(response.text)
    return output_text
