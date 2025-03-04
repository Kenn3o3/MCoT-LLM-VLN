# testapi.py
import os
from openai import OpenAI

def test_api():
    # Initialize client
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )

    try:
        # Simple text-only test
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{
                "role": "user",
                "content": "Answer in one word: What is the capital of France?"
            }],
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip()
        return f"API test passed! Response: '{answer}' (Expected 'Paris')"
        
    except Exception as e:
        return f"API test failed: {str(e)}"

if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("Error: Set DASHSCOPE_API_KEY environment variable first")
        print("Example: export DASHSCOPE_API_KEY='your-api-key-here'")
        exit(1)
    
    print("Testing API connection...")
    result = test_api()
    print(result)