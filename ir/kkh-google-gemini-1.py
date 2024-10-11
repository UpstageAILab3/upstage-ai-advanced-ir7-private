import pathlib
import textwrap
import os
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

content = "헬륨이 다른 원소들과 반응을 잘 안하는 이유는?"
answer = "희귀한 기체인 헬륨, 네온, 아르곤, 크립톤, 크세논, 라돈은 다른 원소들과 거의 반응하지 않습니다. 이는 최외각 에너지 준위가 완전하게 채워져 있기 때문입니다. 이러한 특성으로 인해 이러한 희귀 기체들은 안정하고 비활성인 성질을 가지고 있습니다. 이들은 화학 반응에서 거의 관여하지 않으며, 다른 원소들과 결합하여 화합물을 형성하지 않습니다. 이러한 특징은 희귀 기체들을 다양한 산업 분야에서 사용할 수 있게 만들어줍니다. 예를 들어, 헬륨은 기체 냉매로 사용되며, 네온은 광고 표시판에 사용됩니다. 아르곤은 용접 작업에 사용되고, 크세논은 높은 휘도를 가진 조명 장치에 사용됩니다. 이러한 희귀 기체들은 우리 일상 생활에서도 다양한 용도로 활용되고 있습니다."


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

file_path='./ex-key/google-aistudio-helpotcreator-key.txt'
def load_api_key(file_path):
    try:
        with open(file_path, 'r') as file:
            api_key = file.readline().strip()
        if not api_key:
            raise ValueError("API key is empty.")
        os.environ["OPENAI_API_KEY"] = api_key
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {file_path} not found.")
    except ValueError as e:
        raise ValueError(f"Error: {e}")

GEMINI_API_KEY = load_api_key(file_path)
genai.configure(api_key=GEMINI_API_KEY)

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(f"""
- 질문: {content}\n
\n
##################\n
\n
- 답변: {answer}\n
\n
##################\n

위 질문에 대한, 위 답변이 얼마나 적절한지 100점 중에 몇 점인지 소수점 네 자리까지로만 표현해주세요.
예를 들면, 88.2453점, 24.1134점 등등으로 표현해주세요.
점수만 표현해주고, 다른 내용은 적지 말아주세요.
""")

print('=== response.text ===\n')
print(response.text)
print('=====================')
print('=== response.candidates ===\n')
print(response.candidates)
print('=====================')