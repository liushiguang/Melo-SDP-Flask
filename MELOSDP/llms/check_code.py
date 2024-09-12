import requests
import json
import re
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


def check_code(programming_language, code_segment):
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": f"""Next, you will play the role of a programming expert, check the defects of the code, locate them and give an explanation of the error. First check the code segment, use $ to surround the place where the code defects are detected, and use <code> </code> to mark the code segment when answering. Then give an explanation of the marked error code, explain the cause of the error, and use <text></text> to mark the explanation text segment. No need to reply to anything else
                
Q: Programming language: c++
Code: if a>5 {{print("big");}}
A: <code>if $a>5$ {{$print("big")$;}}</code> \n <text>C++'s if statement requires the condition to be placed in parentheses () and the statement block to be placed in curly braces {{}}. In addition, the print function in C++ should be std::cout. </text>

Q: Programming language: {programming_language}
Code: {code_segment}
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
    return unicode_to_symbols(extract_content(response.text))


# Example usage
programming_language = "python"
code_segment = 'if a>5 \n  print("hello world")'

result = check_code(programming_language, code_segment)
print(result)
