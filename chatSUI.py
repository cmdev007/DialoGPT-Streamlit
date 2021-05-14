from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
import streamlit as st
from load_css import local_css

st.markdown(f"<div style='font-size: 50px;'><center><b>Talk with Jarvis</b></center></div>", unsafe_allow_html=True)
st.markdown("---")

local_css("style.css")

@st.cache(allow_output_mutation=True)
def Loader():
    f = open("step.txt", "w")
    f.write("0")
    f.close()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = Loader()
# Let's chat for 5 lines
f = open("step.txt")
step = int(f.read())
f.close()

# encode the new user input, add the eos_token and return a tensor in Pytorch
hInput = st.text_input("Type:")
if st.button("OK"):
    HI = f"<div><span class='highlight blue'><span class='bold'>You: </span>{hInput}</span></div>"
    st.markdown(HI, unsafe_allow_html=True)
    new_user_input_ids = tokenizer.encode(hInput + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    # bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    AIOut = ("{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

    AI = f"<div><span class='highlight red'><span class='bold'>AI: </span>{AIOut}</span></div>"
    st.write("")
    st.markdown(AI, unsafe_allow_html=True)

    if step == 4:
        step = 0
    else:
        step+=1

    f = open("step.txt","w")
    f.write(str(step))
    f.close()
