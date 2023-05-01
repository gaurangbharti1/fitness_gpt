# Bring in deps
import os 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun

apikey = str(st.secrets['OPENAI_KEY'])
os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 PHATGPT')
prompt = st.text_input('Write your squat mistakes', placeholder="Input should look something like: 'butt wink, knee collapse, hip shift'") 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write about some of the causes of {topic} while squatting'
)

script_template = PromptTemplate(
    input_variables = ['title', 'duckduckgo_search'], 
    template='write some tips on how to improve {title} while squatting while leveraging {duckduckgo_search}'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

search = DuckDuckGoSearchRun()
# Show stuff to the screen if there's a prompt
if prompt: 
    print("PROMPT: " + prompt)
    title = title_chain.run(prompt)
    duckduckgo = search.run(f"how to improve {prompt}") 
    script = script_chain.run(title=prompt, duckduckgo_search=duckduckgo)
    
    st.subheader(f"Some of the causes of {prompt}:")
    st.write(title) 
    st.subheader("Tips on improving")
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('DuckDuckGo Research'): 
        st.info(duckduckgo)