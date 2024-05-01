import json
from urllib import request


def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)


if __name__ == '__main__':
    with open('assets/workflow_api.json', 'r') as f:
        prompt_text = f.read()
        prompt = json.loads(prompt_text)
        queue_prompt(prompt)