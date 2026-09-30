from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Loading Env contians Hugging face key 
load_dotenv()

# Hugging Face Model Free Model 
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text_generation",
    max_new_tokens=200,
    provider="featherless-ai",
    temperature=0.7,
)

# Model using 
model = ChatHuggingFace(llm=llm)

# Giving Chat history 
chat_history = [
    SystemMessage(content='You are a helpful AI Assistant')
]

# Chatbot Interface with appending chat History
while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)