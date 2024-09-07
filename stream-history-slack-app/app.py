import os
from dotenv import load_dotenv
# slack
from slack_bolt import App  
from slack_bolt.adapter.socket_mode import SocketModeHandler
# openai
import re
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from datetime import timedelta
from langchain_community.chat_message_histories import MomentoChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import time
from typing import Any

import json
import logging
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

#  logger
SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(
  format="%(asctime)s [%(levelname)s] %(message)s",
  level=logging.INFO
)
logger = logging.getLogger(__name__)

# momento

CHAT_UPDATE_INTERVAL_SEC = 1
# 환경 변수 로드  
load_dotenv()

# 봇 토큰과 소켓 모드 핸들러를 사용하여 앱을 초기화
app = App(
  signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
  token=os.environ.get("SLACK_BOT_TOKEN"),
  process_before_response=True
)


# # 이벤트 리스너 함수 추가
# # 해당 앱을 멘션하면 응답 
# # 멘션의 스레드에 답글을 달 수 있도록 thread_ts를 사용
# @app.event("app_mention")
# def handle_mention(event, say):
#   user = event["user"]
#   thread_ts = event["ts"]
#   say(thread_ts=thread_ts, text=f"Hello <@{user}>!")

# 위 함수를 참고해서 ChatOpenAI를 사용하여 응답 생성
# @app.event("app_mention")
def handle_mention(event, say):
  # 스레드에 답글을 달 수 있도록
  channel = event["channel"]
  thread_ts = event["ts"]
  message = re.sub("<@.*>", "" , event["text"])
  
  #  Momento 히스토리 클라이언트 생성
  id_ts = event["ts"]
  if "thread_ts" in event:
    id_ts = event["thread_ts"]
  
  result = say("\n\nTyping...", thread_ts=thread_ts)
  ts = result["ts"]
   
  history = MomentoChatMessageHistory.from_client_params(
    id_ts,
    os.environ["MOMENTO_CACHE"],
    timedelta(hours=int(os.environ["MOMENTO_TTL"]))
  )
  
  #  프롬프트 템플릿 구성
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", "You are a good assistant."),
      (MessagesPlaceholder(variable_name="chat_history")),
      ("user", "{input}"),
    ]
  )

  #  llm 설정
  callback = SlackStreamingCallBackHandler(channel=channel, ts=ts)
  
  llm = ChatOpenAI(
    model_name=os.environ["OPENAI_API_MODEL"],
    temperature=os.environ["OPENAI_API_TEMPERATURE"],
    streaming=True,
    callbacks=[callback]
  )
  
  chain = prompt | llm | StrOutputParser()
  
  ai_message = chain.invoke(
    {"input": message, "chat_history": history.messages},
  )
  
  history.add_user_message(message)
  history.add_ai_message(ai_message)

  
#  응답 스트림을 수신하는 CallbackHandler 클래스 정의
class SlackStreamingCallBackHandler(BaseCallbackHandler):
  last_send_time = time.time()
  message = ""
  
  # 클래스 초기화
  def __init__(self, channel, ts):
    self.channel = channel
    self.ts = ts
    self.interval = CHAT_UPDATE_INTERVAL_SEC
    self.update_count = 0
  
  # 새 토큰을 수신할 때마다 메시지에 추가
  def on_llm_new_token(self, token:str, **kwargs: Any) -> None:
    self.message += token
    
    now = time.time()
    
    # 간격이 1초 이상이면 메시지를 보냄
    if now - self.last_send_time > CHAT_UPDATE_INTERVAL_SEC:
      app.client.chat_update(
        channel=self.channel,
        ts=self.ts,
        text=f"{self.message}\n\nTyping...",
      )
      self.last_send_time = now
      self.update_count += 1
      
      if self.update_count / 10 > self.interval:
        self.interval = self.interval * 2
        
  # LLM이 끝나면 최종 결과를 메시지에 추가
  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
    message_context = "OpenAI API에서 생성되는 정보는 부적확하거나 부적절할 수 있으며, 우리의 견해를 나타내지 않습니다."
    
    message_blocks = [
      {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},
      {"type": "divider"},
      {
        "type":"context",
        "elements":[
          {
            "type":"mrkdwn",
            "text": message_context
          }
        ]
      }
    ]
    # 최종 결과를 메시지에 추가
    app.client.chat_update(
      channel=self.channel,
      ts=self.ts,
      text=self.message,
      blocks=message_blocks
    )

def just_ack(ack):
  ack()

app.event("app_mention")(ack=just_ack, lazy=[handle_mention])



def handler(event, context):
  logging.info(f"event: {json.dumps(event)}")
  header = event["headers"]
  logger.info(json.dumps(header))  
  
  if "x-slack-retry-num" in header:
    logger.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
    
    return 200
  
  slack_handler = SlackRequestHandler(app=app)
  return slack_handler.handle(event, context)

if __name__ == "__main__":
  SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()
  
