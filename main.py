from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

agent = create_csv_agent(
    ChatOpenAI(),
    "data/final_cocktails.csv",
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True
)

history = ""
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.load_local("history", embeddings, allow_dangerous_deserialization=True)
stored_data = [doc.page_content for doc in faiss_index.docstore._dict.values()]
for element in stored_data:
    history += element

while True:
    user_input = input("Enter your message (type \"quit\" to quit): ")

    if user_input.strip() == "quit" or user_input == "":
        break

    if "my favorite" in user_input.lower():
        history += user_input + "; "
        faiss_index.add_texts(history)
        faiss_index.save_local("history")

    print(agent.invoke(user_input + " You can also use this info if needed: " + history)["output"] + "\n")