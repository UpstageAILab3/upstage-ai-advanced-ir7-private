import os
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown


class Gemini:
    def __init__(self, api_key_path, model="gemini-1.5-flash"):
        self.api_key_path = api_key_path
        self.model = None
        self.load_api_key()
        self.configure_model(model)

    def load_api_key(self):
        """Load API key from a specified file."""
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.readline().strip()
            if not api_key:
                raise ValueError("API key is empty.")
            os.environ["OPENAI_API_KEY"] = api_key
            return api_key
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {self.api_key_path} not found.")
        except ValueError as e:
            raise ValueError(f"Error: {e}")

    def configure_model(self, model):
        """Configure the generative model."""
        genai.configure(api_key=os.environ["OPENAI_API_KEY"])
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(m.name)
        self.model = genai.GenerativeModel(model)

    @staticmethod
    def to_markdown(text):
        """Convert text to Markdown format."""
        text = text.replace("•", "  *")
        return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

    def prompt_1(self, content):
        response = self.model.generate_content(f"""
        - 내용: {content}\n
        \n
        ##################\n

        위 내용에 대해, 요약해 주세요.
        최대 500글자를 넘지 않도록 해주세요.
        요약 내용만 적어주고, 다른 내용은 적지 말아주세요.
        """)

        display(response.text)
        # display(response.candidates)
        return response.text

    def prompt_2(self, content):
        response = self.model.generate_content(f"""
        - 내용: {content}\n
        \n
        ##################\n

        위 내용에 대해, 하나의 질문으로 요약해 주세요.
        최대 500글자를 넘지 않도록 해주세요.
        요약 내용만 적어주고, 다른 내용은 적지 말아주세요.
        """)

        display(response.text)
        # display(response.candidates)
        return response.text

    def prompt_3(self, content, answer):
        """Generate content score for the provided question and answer."""
        response = self.model.generate_content(f"""
        - 질문: {content}\n
        \n
        ##################\n
        \n
        - 답변: {answer}\n
        \n
        ##################\n

        위 질문에 대한, 위 답변이 얼마나 적절한지 100점 중에 몇 점인지 소수점 네 자리까지로만 표현해주세요.
        예를 들면, 88.2453, 24.1134 등등으로 표현해주세요.
        점수만 표현해주고, 다른 내용은 적지 말아주세요.
        """)

        display(response.text)
        # display(response.candidates)
        return response.text

def start():
    gemini = Gemini(api_key_path='./ex-key/google-aistudio-helpotcreator-key.txt', model="gemini-1.5-flash")

    # content = "헬륨이 다른 원소들과 반응을 잘 안하는 이유는?"
    # doc = "희귀한 기체인 헬륨, 네온, 아르곤, 크립톤, 크세논, 라돈은 다른 원소들과 거의 반응하지 않습니다."
    # gemini_answer = gemini.prompt_3(content, doc)
    # 답변: 50.0000

    # content = "우유 알레르기/불내증이 의심되는 영유아와 어린이들은 다음과 같은 증상 및 징후를 나타낼 수 있습니다:\n\n1. 피부 발진: 우유 알레르기를 가진 아이들은 피부에 발진이 나타날 수 있습니다. 이는 가려움증과 함께 나타날 수 있으며, 종종 빨간 반점이나 발진이 퍼질 수 있습니다.\n\n2. 소화 장애: 우유 알레르기를 가진 아이들은 소화 장애를 겪을 수 있습니다. 이는 구토, 설사, 복통 등의 증상으로 나타날 수 있으며, 심한 경우에는 탈수의 위험도 증가할 수 있습니다.\n\n3. 호흡곤란: 우유 알레르기를 가진 아이들은 호흡곤란을 경험할 수 있습니다. 이는 코막힘, 기침, 숨 가쁨 등의 증상으로 나타날 수 있으며, 심한 경우에는 천식 발작을 유발할 수 있습니다.\n\n4. 알레르기 비염: 우유 알레르기를 가진 아이들은 알레르기 비염을 겪을 수 있습니다. 이는 코막힘, 재채기, 코주부 등의 증상으로 나타날 수 있으며, 가려움증과 함께 나타날 수도 있습니다.\n\n5. 아나필락시스: 심각한 경우에는 우유 알레르기로 인해 아나필락시스라는 심각한 반응이 나타날 수 있습니다. 이는 호흡곤란, 혈압 저하, 응급 상황 등을 초래할 수 있으며, 즉각적인 응급 조치가 필요합니다.\n\n위의 모든 항목은 우유 알레르기/불내증이 의심되는 영유아와 어린이들이 나타낼 수 있는 증상 및 징후입니다. 만약 이러한 증상이 나타난다면, 즉시 의료진과 상담하고 적절한 조치를 취해야 합니다."
    # gemini_answer = gemini.prompt_1(content)
    # 답변: 우유 알레르기/불내증이 의심되는 영유아 및 어린이는 피부 발진, 소화 장애, 호흡곤란, 알레르기 비염 등의 증상을 보일 수 있습니다. 특히 심각한 경우에는 아나필락시스라는 심각한 반응이 나타날 수 있으며, 이는 호흡곤란, 혈압 저하, 응급 상황 등을 초래할 수 있습니다. 이러한 증상이 나타난다면 즉시 의료진과 상담하여 적절한 조치를 취해야 합니다.

    content = "user: 이란 콘트라 사건이 뭐야\nassistant: 이란-콘트라 사건은 로널드 레이건 집권기인 1986년에 레이건 행정부와 CIA가 적성국이었던 이란에게 무기를 몰래 수출한 대금으로 니카라과의 우익 성향 반군 콘트라를 지원하면서 동시에 반군으로부터 마약을 사들인 후 미국에 판매하다가 발각되어 큰 파장을 일으킨 사건입니다.\nuser: 이 사건이 미국 정치에 미친 영향은?"
    gemini_answer = gemini.prompt_2(content)
    # 답변: 이란-콘트라 사건은 미국의 정치에 어떤 영향을 미쳤을까요?

if __name__ == "__main__":
    start()
