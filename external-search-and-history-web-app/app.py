import os
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

def create_agent_chain(history):
  # 모델 생성
  chat = ChatOpenAI(
    model_name=os.environ["OPENAI_API_MODEL"],
    temperature=os.environ["OPENAI_API_TEMPERATURE"],
  )
  
  # 에이전트가 사용할 툴 생성
  tools = load_tools(["ddg-search", "wikipedia"])
  
  #  프롬프트
  prompt = hub.pull("hwchase17/openai-tools-agent")
  
  # 메모리
  memory = ConversationBufferMemory(
    chat_memory=history, memory_key="chat_history", return_messages=True
  )
  
  agent = create_openai_tools_agent(chat, tools, prompt)
  
  return AgentExecutor(agent=agent, tools=tools, memory=memory)

st.title("langchain-streamlit-app")

# 채팅 메시지 히스토리
history = StreamlitChatMessageHistory()

for message in history.messages:
  st.chat_message(message.type).write(message.content)

#  사용자 입력
prompt = st.chat_input("What is up?")

# 사용자 입력이 있을 경우
if prompt:
  # 사용자 이모티콘으로 마크다운 표시
  with st.chat_message("user"):
    st.markdown(prompt)
  
  # AI 아이콘으로 마크다운 표시
  with st.chat_message("assistant"):
    callback = StreamlitCallbackHandler(st.container())
    
    agent_chain = create_agent_chain(history)
    response = agent_chain.invoke(
      {"input": prompt},
      {"callback": [callback]},
    )

    st.markdown(response["output"])
  
  
  
  