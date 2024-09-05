# 외부 검색과 히스토리를 바탕으로 응답하는 웹 앱 구축하기

사용 API: OpenAI Completions API
UI: streamlit

## 개발 단계
### 필요한 패키지 설치
```sh
pip install langchain==0.1.14 openai==1.14.3 langchain-openai==0.0.1
```
대화 기록을 저장하기 위해 `ConversationBufferMemory`를 사용한다.

LLM이 duckduckgo 또는 wikipedia에서 검색을 할 수 있도록 tools를 사용한다.

프롬프트는 기존 hub에서 가져온다.

