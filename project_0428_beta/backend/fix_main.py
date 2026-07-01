"""
修复 main.py 中的引号问题
"""
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复模型名称缺少引号的问题
content = content.replace(
    'self.model = os.getenv("OPENAI_MODEL", glm-5.1)',
    'self.model = os.getenv("OPENAI_MODEL", "glm-5.1")'
)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("修复完成！")
