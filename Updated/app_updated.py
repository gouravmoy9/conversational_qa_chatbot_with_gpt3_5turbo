
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

import streamlit as st
from streamlit_chat import message
from utils import *

if 'responses' not in st.session_state:
    st.session_state['responses'] = ['Put in a Query.']

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)



st.title("LangChain Chatbot")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

sys_msg_tp = SystemMessagePromptTemplate.from_template(template='''Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know''')

hum_msg_tp = HumanMessagePromptTemplate.from_template(template='{input}')

prompt_template = ChatPromptTemplate.from_messages([sys_msg_tp, MessagesPlaceholder(variable_name='history'), hum_msg_tp])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
response_con = st.container()
text_con = st.container()

with text_con:
    query = st.text_input('Query: ', key='input')
    if query:
        with st.spinner('thinking...'):
            convo_string = get_conversation_string()
            refined_query = query_refiner(convo_string, query)
            st.subheader('Refined Query:')
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f'Context:\n{context}\n\nQuery:\n{query}')
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_con:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=str(i)+'user')

