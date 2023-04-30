# Bring in deps
import os 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

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
    input_variables = ['title', 'wikipedia_research'], 
    template='write some tips on how to improve {title} while squatting while leveraging {wikipedia_research}'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()
arxiv = ArxivAPIWrapper()
# Show stuff to the screen if there's a prompt
if prompt: 
    print("PROMPT: " + prompt)
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    #with st.expander('Wikipedia Research'): 
        #st.info(wiki_research)