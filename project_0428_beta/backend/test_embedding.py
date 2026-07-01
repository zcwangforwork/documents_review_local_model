"""
测试本地 Ollama Embedding API（OpenAI 兼容接口）
"""
import os
import httpx

API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:11435/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")

def test_embedding():
    """测试 embedding API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": "天很蓝，海很深"
    }

    print(f"测试 API 地址: {API_URL}")
    print(f"请求内容: {payload}")

    try:
        with httpx.Client(timeout=60, trust_env=False) as client:
            response = client.post(API_URL, headers=headers, json=payload)
            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"响应结构: {list(result.keys())}")
                data = result.get("data", [])
                if data:
                    embedding = data[0].get("embedding", [])
                    print(f"成功获取 embedding，长度: {len(embedding)}")
                    print(f"前5个值: {embedding[:5]}")
                    return True
                else:
                    print(f"响应 data 为空: {result}")
            else:
                print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    return False

if __name__ == "__main__":
    success = test_embedding()
    print(f"\n测试结果: {'成功' if success else '失败'}")
