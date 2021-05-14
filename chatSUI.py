import datetime
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle, os, sqlite3
import streamlit as st
from load_css import local_css

st.set_page_config(
    page_title="Jarvis",
    page_icon="👽",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(f"<div style='font-size: 50px;'><center><b>Talk with Jarvis</b></center></div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: 12px;'><center>By <a href='https://github.com/cmdev007/'><span class='highlight green'><span class='bold'>cMDev007</span></span></a></center></div>", unsafe_allow_html=True)

local_css("style.css")

############### SQLITTE ###############

def create_table():
    conn = sqlite3.connect('messages.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS msgdata(unix REAL, datestamp TEXT, uname TEXT, hitext TEXT, aitext TEXT)")
    c.close()
    conn.close()


def data_entry(unix, datestamp, uname, hitext, aitext):
    hitext = hitext.replace("'", "''")
    aitext = aitext.replace("'", "''")
    uname = uname.replace("'", "''")
    conn = sqlite3.connect('messages.db')
    c = conn.cursor()
    c.execute(f"INSERT INTO msgdata VALUES('{unix}', '{datestamp}', '{uname}', '{hitext}', '{aitext}')")
    conn.commit()
    c.close()
    conn.close()

def read_data(uname):
    uname = uname.replace("'", "''")
    conn = sqlite3.connect('messages.db')
    c = conn.cursor()
    c.execute(f"SELECT * FROM msgdata WHERE uname = '{uname}'")
    data = c.fetchall()
    return data

############## SQLITE ################

if "messages.db" not in os.listdir():
    create_table()

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

app_state = st.experimental_get_query_params()
if app_state == {}:
    with st.form(key='columns_in_form'):
        cont0 = st.empty()
        cont0.beta_container()
        col03, col01, col02, col04 = cont0.beta_columns([0.15, 1, 0.20, 0.15])

        nInputobj = col01.empty()
        nInput = nInputobj.text_input("Your name please:")

        contextobj = col02.empty()
        context = contextobj.selectbox("context", ('2', '3', '4', '5'))
        context = int(context) - 1

        submitted = col02.form_submit_button('Submit')



app_state = st.experimental_get_query_params()

if app_state!={}:
    nInput = app_state['nInput'][0]
    context = int(app_state['context'][0])

if nInput != "":
    nInput = nInput.lower().replace(" ", "")
    st.experimental_set_query_params(**{"nInput" : nInput, "context" : str(context)})

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
    col2.write("")
    col2.write("")

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    hInput = col1.text_input("Type:")
    if col2.button("Send"):
        uTime = time.time()
        tStamp = datetime.datetime.fromtimestamp(uTime).strftime('%HH-%MM : %d-%m-%Y')
        HI = f"<div><span class='highlight blue'><span class='bold'>You: </span>{hInput}</span></div>"
        col1.markdown(HI, unsafe_allow_html=True)
        os.system(f'''echo "{nInput} : {hInput}"''')
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
        os.system(f'''echo AI to {nInput} : "{AIOut}"''')

        histData = read_data(nInput)
        for i in range(len(histData)):
            buffData = histData[len(histData)-i-1]
            col1.markdown("---")

            # col1.write("")
            HI = f"<div><span class='highlight blue'><span class='bold'>You: </span>{buffData[3]}</span></div>"
            col1.markdown(HI, unsafe_allow_html=True)

            AI = f"<div><span class='highlight red'><span class='bold'>AI: </span>{buffData[4]}</span></div>"
            col1.write("")
            col1.markdown(AI, unsafe_allow_html=True)

        data_entry(uTime, tStamp, nInput, hInput, AIOut)
        os.system("echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(read_data(nInput))
        os.system("echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        if step >= context+1:
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