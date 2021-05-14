from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle, os
import streamlit as st
from load_css import local_css

st.markdown(f"<div style='font-size: 50px;'><center><b>Talk with Jarvis</b></center></div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: 12px;'><center>By <a href='https://github.com/cmdev007/'><span class='highlight green'><span class='bold'>cMDev007</span></span></a></center></div>", unsafe_allow_html=True)

local_css("style.css")

@st.cache(allow_output_mutation=True)
def Loader():
    f = open("step.txt", "w")
    f.write("0")
    f.close()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

@st.cache(allow_output_mutation=True)
def Namer(name):
    a = name
    return a

tokenizer, model = Loader()

cont0 = st.beta_container()
col03, col01, col02, col04 = cont0.beta_columns([0.15, 1, 0.15, 0.15])

nInput = col01.text_input("Your name please:")
nInput = nInput.lower().replace(" ","")
context = col02.selectbox("context",('2', '3', '4', '5'))
context = int(context)-1

if nInput != "":
    if nInput not in os.listdir():
        os.system(f"mkdir {nInput}")
    st.markdown("---")

    try:
        f = open(f"{nInput}/step.txt")
        step = int(f.read())
        f.close()
    except:
        f = open(f"{nInput}/step.txt", "w")
        f.write("0")
        f.close()

    cont1 = st.beta_container()
    col3, col1, col2, col4 = cont1.beta_columns([0.15, 1, 0.125, 0.15])
    # col2.button("OK")
    # uName = Namer(nInput)
    # st.info(f"Hello {uName}")
    col2.write("")
    col2.write("")

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    hInput = col1.text_input("Type:")
    if col2.button("Send"):
        HI = f"<div><span class='highlight blue'><span class='bold'>You: </span>{hInput}</span></div>"
        col1.markdown(HI, unsafe_allow_html=True)
        os.system(f"echo {nInput} : {hInput}")
        new_user_input_ids = tokenizer.encode(hInput + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        try:
            g = open(f"{nInput}/chat_history_ids", "rb")
            chat_history_ids = pickle.load(g)
            g.close()
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                                      dim=-1) if step > 0 else new_user_input_ids
        except:
            step = 0
            bot_input_ids = new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        g = open(f"{nInput}/chat_history_ids", "wb")
        pickle.dump(chat_history_ids, g)
        g.close()
        # pretty print last ouput tokens from bot
        AIOut = (
            "{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

        AI = f"<div><span class='highlight red'><span class='bold'>AI: </span>{AIOut}</span></div>"
        col1.write("")
        col1.markdown(AI, unsafe_allow_html=True)
        os.system(f"echo AI : {AIOut}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if step >= context:
            step = 0
        else:
            step += 1

        f = open(f"{nInput}/step.txt", "w")
        f.write(str(step))
        f.close()


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)