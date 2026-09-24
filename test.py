import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen3.5:9b",
    "prompt": "Explain how Python decorators work.",
    "stream": False
}

response = requests.post(url, json=payload)
response.raise_for_status()

data = response.json()
print(data["response"])
