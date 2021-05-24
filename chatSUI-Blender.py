import datetime
import time
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
import torch
#from streamlit_lottie import st_lottie
import pickle, os, sqlite3
import streamlit as st
from load_css import local_css
import json

st.set_page_config(
    page_title="Jarvis",
    page_icon="👽",
    layout="centered",
    initial_sidebar_state="auto",
)

placeTitle = st.empty()
placeTitle.markdown(f"<div style='font-size: 50px;;color:grey;font-family:orbitron;'><center><b>Talk with Jarvis</b></center></div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: 12px;'><center>By <a href='https://github.com/cmdev007/'><span class='highlight green'><span class='bold'>Preet</span></span></a></center></div>", unsafe_allow_html=True)

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
    mname = 'facebook/blenderbot-90M'
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
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
    # f = open("animations/face-scan.json")
    # jData = json.load(f)
    # f.close()
    # st_lottie(jData, quality="high",width=100)


app_state = st.experimental_get_query_params()

if app_state!={}:
    nInput = app_state['nInput'][0]
    context = int(app_state['context'][0])

if nInput != "":
    placeTitle.markdown(f"<div style='font-size:30px;color:grey;font-family:orbitron;'>\
    <center><b>Talk with Jarvis</b></center></div>",
                        unsafe_allow_html=True)

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
        # HI = f"<div><span class='highlight blue'><span class='bold'>You: </span>{hInput}</span></div>"
        # col1.markdown(HI, unsafe_allow_html=True)
        os.system(f'''echo "{nInput} : {hInput}"''')
        inputs = tokenizer([hInput], return_tensors='pt')
        # inputs.pop("token_type_ids")
        reply_ids = model.generate(**inputs)

        AIOut = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

        # AI = f"<div><span class='highlight red'><span class='bold'>AI: </span>{AIOut}</span></div>"
        # col1.write("")
        # col1.markdown(AI, unsafe_allow_html=True)
        os.system(f'''echo AI to {nInput} : "{AIOut}"''')

        data_entry(uTime, tStamp, nInput, hInput, AIOut)
        os.system("echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        if step >= context+1:
            step = 0
        else:
            step += 1

        f = open(f"{nInput}/step.txt", "w")
        f.write(str(step))
        f.close()

    cont2 = st.beta_container()
    col23, col21, col22 = cont2.beta_columns([0.13, 1, 0.10])

    histData = read_data(nInput)
    for i in range(len(histData)):
        buffData = histData[len(histData) - i - 1]
        col21.markdown("---")

        # col1.write("")
        HI = f"<div><span class='highlight blue'><span class='bold'>You: </span>{buffData[3]}</span></div>"
        col21.markdown(HI, unsafe_allow_html=True)

        AI = f"<div align='right'><span class='highlight red'><span class='bold'>AI: </span>{buffData[4]}</span></div>"
        col21.write("")
        col21.markdown(AI, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
