from openai import OpenAI
client = OpenAI()

def openai_llm(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
