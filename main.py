import hmac
import streamlit as st


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()

import openai
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import pinecone
import pandas as pd

openai_api_key = st.secrets.openai.api_key
openai_model = st.secrets.openai.model
openai_embedding_model = st.secrets.openai.embedding_model
pinecone_api_key = st.secrets.pinecone.api_key
pinecone_environment = st.secrets.pinecone.environment
pinecone_index_name = st.secrets.pinecone.index_name
pinecone_text_field = st.secrets.pinecone.text_field

openai.api_key = openai_api_key
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    model=openai_model
)
embed_model = OpenAIEmbeddings(model=openai_embedding_model, openai_api_key=openai_api_key)
pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)
index = pinecone.Index(pinecone_index_name)
vectorstore = Pinecone(
    index, embed_model.embed_query, pinecone_text_field
)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def augment_prompt(query: str):
    # get top 5 results from knowledge base
    results = vectorstore.similarity_search(query, k=5)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Используя приведенные ниже транскрибации интервью, ответьте на запрос.

    Транскрибации интервью:
    {source_knowledge}

    Запрос: {query}"""
    return augmented_prompt, results

system_message = """
Ты старший продуктовый исследователь.
Твоя задача предоставить инсайты, исходящие из контекста приведенных транскрибаций интервью.
Давай разберемся с этим шаг за шагом, чтобы быть уверенными, что у нас есть правильный ответ.
"""
search_result_template = """
**Ссылка:** {url}\n\n
**Контент:**\n
{page_content}
"""
prompt_messages = [
    SystemMessage(content=system_message.strip().replace("\n", " "))
]

st.title("💬 Чатбот: ТЖ ремонт")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content=system_message.strip().replace("\n", " "))
    ]
if "combined_data" not in st.session_state:
    st.session_state["combined_data"] = []
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []

n = 0
for msg in st.session_state.messages:
    if msg.type != "system":
        st.chat_message(msg.type).write(msg.content)
        if msg.type == "ai":
            n_divider = 0
            with st.expander("Источники"):
                for search_result in st.session_state.search_results[n]:
                    n_divider += 1
                    st.write(search_result_template.format(
                        url=search_result.metadata["source"],
                        page_content=search_result.page_content
                    ))
                    if n_divider != 5:
                        st.divider()
            n += 1

if prompt := st.chat_input(placeholder="Ваше сообщение"):
    st.chat_message("human").write(prompt)
    augmentated_prompt, search_results = augment_prompt(prompt)
    modified_prompt = HumanMessage(content=augmentated_prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    res = chat(prompt_messages + [modified_prompt])
    st.session_state.messages.append(AIMessage(content=res.content))
    st.chat_message("ai").write(res.content)
    st.session_state["combined_data"].append({
        "question": prompt,
        "answer": res.content,
        "urls": "\n".join([x.metadata["source"] for x in search_results])
    })
    st.session_state["search_results"].append(search_results)
    n_divider = 0
    with st.expander("Источники"):
        for search_result in st.session_state.search_results[-1]:
            n_divider += 1
            st.write(search_result_template.format(
                url=search_result.metadata["source"],
                page_content=search_result.page_content
            ))
            if n_divider != 5:
                st.divider()

csv = convert_df(pd.DataFrame(st.session_state["combined_data"]))
st.download_button(
    label="Скачать переписку в виде таблицы",
    data=csv,
    file_name='tinkoff-gpt-chatting.csv',
    mime='text/csv',
)