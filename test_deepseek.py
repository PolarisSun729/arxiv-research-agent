from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)

def chat_with_reasoner(prompt):
    """
    使用deepseek-reasoner模型进行对话
    """
    response = client.chat.completions.create(
        # model="deepseek-reasoner",
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # 获取推理过程和最终答案
    # reasoning = response.choices[0].message.reasoning_content
    reasoning = 'ABC'
    answer = response.choices[0].message.content
    
    return reasoning, answer

def main():
    # 测试问题
    question = "计算23和45的和，并解释计算过程。"
    
    try:
        reasoning, answer = chat_with_reasoner(question)
        print("推理过程：")
        print(reasoning)
        print("\n最终答案：")
        print(answer)
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    main()
