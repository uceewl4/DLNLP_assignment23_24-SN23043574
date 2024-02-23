import re
from transformers import pipeline
from spacy_streamlit import visualize_spans
import spacy_transformers
import nltk
import transformers
from sentence_transformers import SentenceTransformer, util
import langchain_community
from transformers import pipeline
import openai
from langchain import LLMChain, PromptTemplate
from langchain import HuggingFaceHub
import requests
from streamlit_chat import message
from transformers import (
    AutoModelForSequenceClassification,
    LongformerForSequenceClassification,
    LongformerTokenizer,
)
import os
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import spacy
import spacy_experimental
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import sentencepiece
import sacremoses
import requests
import streamlit as st
import annotated_text
from annotated_text import annotated_text
import smtplib
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
from email.header import Header
from email.mime.text import MIMEText
from streamlit_option_menu import option_menu
from spacy import displacy

# page configuration
st.set_page_config(page_title="NLPITS")  # system name
st.markdown(
    """
    <style>
    body {
        primary-color: #FF4B4B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# There are 7 main columns in the sidebar:
# 1. Welcome is designed to give basic guidelines and instructions for new users of the system
# 2. Chatbot is a real-time com-munication frame like ChatGPT, where users can input message at sidebar and get interactive dialogue by stored session
# message and invocation of ChatOpenAI.
# 3. Sentiment&Intent is designed for online analysis of sentiment and intent for a given sentence.
# 4. Sentence analysis is designed for comprehensive analysis for given sentences from
# NEM, POS tagging, co-reference resolution and depsndency parsing.
# 5. Machine comprehension is designed as some implementable tools used commonly in our daily
# life, including Question answering for passage comprehension, Text summarization, Machine translation and Semantic
# textual similarity for essay similarity checking.
# 6. Story telling includes two parts of image to text and text to story.
# 7. Help section is designed for user to provide suggestion for NLPITS developers where messages will be
# sent automatically by email.

with st.sidebar:  # sidebar of the system, NLP interactive tasks system NLPITS
    choose = option_menu(
        "NLPITS",
        [
            "Welcome",
            "Chatbot",
            "Sentiment&Intent",
            "Sentence analysis",
            "Machine comprehension",
            "Story telling",
            "Help",
        ],
        menu_icon="android",
        icons=[
            "compass",
            "chat",
            "emoji-smile",
            "text-paragraph",
            "translate",
            "book",
            "question-circle",
        ],
        default_index=0,
    )

# welcome
if choose == "Welcome":  # instruction page
    st.title("🎊 Welcome to NLPITS!")
    st.write(
        "Hi, welcome to the Natural Language Processing Interactive Tasks System (NLPITS). This is the instruction page for NLPITS"
        + ", an integrated natural language processing kit. New users of this app can refer to the "
        + "following sections as guidelines. 👇"
    )

    st.header("Chatbot")
    with st.expander("See details", expanded=True):
        st.write(
            "This is where you can make real-time communication with a Chatgpt-like assistant."
        )
        st.subheader("👉 Want to talk to our chatbot?" " -- See our *Chatbot* section.")
        st.write(
            "• The **Chatbot** will respond to your required messages *:blue[at sidebar]* in a communicative way!"
        )

    st.header("Sentiment&Intent")
    with st.expander("See details", expanded=True):
        st.write(
            "This is the place for determining the sentiment and intent of your sentences."
        )
        st.subheader(
            "👉 Want to check your sentiment?" "-- *Sentiment classification* helps."
        )
        st.write(
            "• Sentences can be fed to further analyze latent sentiment and emotions as **positive, negative and neutral**."
        )

        st.subheader("🙌 Additional tool for intent recognition!")
        st.write("• Input your sentence to analyze embedded intent and purposes.")

    st.header("Sentence analysis")
    with st.expander("See details", expanded=True):
        st.write(
            "This is the place for analyzing sentence structure with name entity recognition, part of speech tagging, coreference resolution and dependency parsing."
        )
        st.subheader("👉 Want to recognize name entity?" "-- *NER* helps.")
        st.write(
            "• Figure out name entity in your sentence with **name, person, location, etc.**"
        )

        st.subheader("🙌 Assign syntax meaning for part of speech tagging!")
        st.write("• See our *POS-tagging section* to analyze syntax structure.")

        st.subheader("👉 Want to connect coreference resolution?")
        st.write("• Click for this section to analyze coreferences in the text.")

        st.subheader("🙌 Hierarchy for dependency parsing!")
        st.write(
            "• See this section to parse dependency embedded in sentences and words."
        )

    st.header("Machine comprehension")
    with st.expander("See details", expanded=True):
        st.write(
            "This is the place for performing machine comprehension tasks like Q&A, text summarization, translation and semantic similarity analysis."
        )
        st.subheader("👉 Want to conduct question answering?")
        st.write(
            "• Click on *Question answering* section to figure out answer from given passages."
        )

        st.subheader("🙌 Summrize your articles as abstract!")
        st.write("• See our *Text summarization* to conclude your article.")

        st.subheader("👉 Want to translate between multiple languages?")
        st.write(
            "• Click for *Machine translation* to translate between **Chinese and English**."
        )

        st.subheader("🙌 Check similarity for your essay!")
        st.write(
            "• See *Semantic textual similarity* section to compare your essays with others."
        )

    st.header("Story telling")
    with st.expander("See details", expanded=True):
        st.write(
            "This is the place for multimodal conversion between **text, audio and image** for story telling."
        )
        st.subheader("👉 Want to convert your image into text?")
        st.write("• Images with .JPG format can be described as text automatically!")

        st.subheader("🙌 Generate story for background text!")
        st.write(
            "• Create new story based on your required scenario and convert it into audio teller."
        )

    st.header("Help")
    with st.expander("See details", expanded=True):
        st.write(
            "Contact **NLPITS Developer Team** if you have any problems or suggestions. Glad to see your "
            + "contribution."
        )

# chatbot
elif choose == "Chatbot":
    chat = ChatOpenAI(
        openai_api_key="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"
    )

    if "messages" not in st.session_state:  # message history
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))

    messages = st.session_state.get("messages", [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + "_user")
        else:
            message(msg.content, is_user=False, key=str(i) + "_ai")

# sentiment and intent
elif choose == "Sentiment&Intent":
    task = option_menu(
        None,
        ["Sentiment classification", "Intent recognition"],
        icons=["hand-thumbs-up", "heart"],
        default_index=0,
        orientation="horizontal",
    )

    if task == "Sentiment classification":
        input = st.text_input("Please input your sentiment sentence")
        if input != "":
            with st.spinner("Wait for it..."):
                sentiment_task = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                )
                res = sentiment_task(str(input))[0]["label"]
                print(res)
            if res == "positive":
                st.success("Your sentiment is positive.", icon="😍")
            elif res == "negative":
                st.error("Your sentiment is negative.", icon="😔")
            else:  # neutral
                st.info("Your sentiment is neutral", icon="😐")

    elif task == "Intent recognition":
        # cpu version
        input = st.text_input("Please input your sentence")
        with st.spinner("Wait for it..."):
            intent_task = pipeline(
                "text-classification", model="Falconsai/intent_classification"
            )
            res = intent_task(input)[0]["label"]
        if input != "":
            st.success(f"Your sentiment is {res}")

# sentence analysis
elif choose == "Sentence analysis":
    tab1, tab2, tab3, tab4 = st.tabs(
        ["NER", "POS tagging", "Coreference resolution", "Dependency parsing"]
    )

    with tab1:
        text = st.text_area("Text to analyze", key="NER")
        with st.spinner("Wait for it..."):
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            ent_html = displacy.render(doc, style="ent", jupyter=False)
        st.markdown(ent_html, unsafe_allow_html=True)

    with tab2:
        text = st.text_area("Text to analyze", key="tab2")
        with st.spinner("Wait for it..."):
            nltk.download("averaged_perceptron_tagger")
            word = nltk.word_tokenize(text)
            res = nltk.pos_tag(word)
        annotated_text(res)
        with st.expander("See parse of speech tagging information."):
            with open("B/nltk._upenn_tagset.txt") as f:
                doc = f.readlines()
                for index, i in enumerate(doc):
                    if ":" in i:
                        st.write(i)
    # # version compatibility
    # with tab3:
    #     # python3 -m spacy download en_core_web_sm
    #     text = st.text_area("Text to analyze", key="tab3")
    #     if text != "":
    #         with st.spinner("Wait for it..."):
    #             # os.system("python3 -m spacy download en_core_web_sm")
    #             # os.system(
    #             #     "pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl"
    #             # )
    #             nlp = spacy.load("en_coreference_web_trf")
    #             doc = nlp(text)
    #         # res = doc.spans["coref_clusters_1"]
    #         # st.success("")
    #         st.text_area(
    #             "Coreference resolution",
    #             f"{str(list(doc.spans.values()))[1:-1]}",
    #             key="res",
    #         )
    #         print(doc.spans)

    # with tab4:
    #     text = st.text_area("Text to analyze", key="DP")
    #     with st.spinner("Wait for it..."):
    #         nlp = spacy.load("en_core_web_sm")
    #         doc = nlp(text)
    #         dep_svg = displacy.render(doc, style="dep", jupyter=False)
    #     st.image(dep_svg, use_column_width=True)

# machine comprehension
elif choose == "Machine comprehension":
    task = option_menu(
        None,
        [
            "Question answering",
            "Text summarization",
            "Machine translation",
            "Semantic textual similarity",
        ],
        icons=["blockquote-right", "body-text", "google", "check"],
        default_index=0,
        orientation="horizontal",
    )

    if task == "Question answering":
        passage = st.text_area("Please input your passage")
        question = st.text_input("Please input your question")
        if passage != "" and question != "":
            with st.spinner("Wait for it..."):
                qa = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2",
                    tokenizer="deepset/roberta-base-squad2",
                )
                answer = qa(question, passage)["answer"]
            res = st.text_area("Answer", answer, key="answer")

    elif task == "Text summarization":
        article = st.text_area("Please input your article")
        if article != "":
            with st.spinner("Wait for it..."):
                model = T5ForConditionalGeneration.from_pretrained("t5-small")
                tokenizer = T5Tokenizer.from_pretrained("t5-small")
                t5_input_text = "summarize: " + article
                tokenized_text = tokenizer.encode(
                    t5_input_text, return_tensors="pt", max_length=1024
                )
                summary_ids = model.generate(
                    tokenized_text, min_length=30, max_length=512
                )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            res = st.text_area("Summary", summary, key="summary")

    elif task == "Machine translation":
        col1, col2 = st.columns(2)
        with col1:
            la_from = st.selectbox("from", tuple(["Chinese", "English"]))
            original = st.text_area("Please input your sentences", height=600)
        with col2:
            la_dest = st.selectbox("dest", tuple(["Chinese", "English"]))
            with st.spinner("Wait for it..."):
                if original != "":
                    if la_from == "English" and la_dest == "Chinese":
                        pipe = pipeline(
                            "translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh"
                        )
                        output = pipe(original)[0]["translation_text"]
                    elif la_from == "Chinese" and la_dest == "English":
                        pipe = pipeline(
                            "translation_zh_to_en", model="Helsinki-NLP/opus-mt-zh-en"
                        )
                        output = pipe(original)[0]["translation_text"]
                    else:
                        output = original
                    res = st.text_area("Translation", output, height=600)

    elif task == "Semantic textual similarity":
        col1, col2 = st.columns(2)
        with col1:
            corpus = st.text_area("Please input your essay", height=400)
        with col2:
            sentence = st.text_area("Please input your sentence", height=100)
            with st.spinner("Wait for it..."):
                if corpus != "" and sentence != "":
                    corpus = str(corpus).split(".")[:-1]
                    model = SentenceTransformer(
                        "sentence-transformers/all-mpnet-base-v2"
                    )
                    corpus_embedding = model.encode(corpus, show_progress_bar=True)
                    query_embedding = model.encode(sentence)
                    search = util.semantic_search(query_embedding, corpus_embedding)[0]
                    for item in search:
                        st.write(
                            str(round(item["score"] * 100, 2))
                            + "% "
                            + str(corpus[item["corpus_id"]])
                            + "\n"
                        )

# story telling
elif choose == "Story telling":
    task = option_menu(
        None,
        ["Image to text", "Text to story"],
        icons=["card-image", "file-music"],
        default_index=0,
        orientation="horizontal",
    )

    if task == "Image to text":
        uploaded_image = st.file_uploader(
            "Choose an image for captioning.", accept_multiple_files=False
        )

        if uploaded_image != None:
            st.download_button(
                f"Download {uploaded_image.name}", uploaded_image, mime="image/JPG"
            )
            # convert to image and store it
            bytes_data = uploaded_image.getvalue()  # to read file as bytes
            bytes_stream = BytesIO(bytes_data)  # convert bytes into stream
            user_img = Image.open(bytes_stream)
            imgByteArr = BytesIO()
            user_img.save(imgByteArr, format("PNG"))
            imgByteArr = imgByteArr.getvalue()
            if not os.path.exists("Outputs/interface/"):
                os.makedirs("Outputs/interface/")
            with open("Outputs/interface/img_to_text.png", "wb") as f:
                f.write(imgByteArr)

            API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
            headers = {"Authorization": "Bearer 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"}
            with open("Outputs/interface/img_to_text.png", "rb") as f:
                data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)
            st.success(response.json()[0]["generated_text"])

    elif task == "Text to story":
        scenario = st.text_input("Please describe your scenario")
        if scenario != "":
            with st.spinner("Wait for it..."):
                repo_id = "tiiuae/falcon-7b-instruct"
                hf_token = "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"
                llm = langchain_community.llms.HuggingFaceHub(
                    huggingfacehub_api_token=hf_token,
                    repo_id=repo_id,
                    verbose=False,
                    model_kwargs={"temperature": 0.1, "max_new_tokens": 1500},
                )
                template = """you are a story teller. 
                            you get a scenario as an input text, 
                            and generates a short story out of it.
                            Context: {scenario}
                            Story:
                            """
                prompt = PromptTemplate(template=template, input_variables=["scenario"])
                chain = LLMChain(prompt=prompt, llm=llm)
                story = chain.predict(scenario=scenario)
                story = story.split(":")[-1].strip()
                story = re.sub("[^a-zA-Z.,;]", " ", story)
                story = " ".join(story.split())
                story = re.sub(r"^\s+", "", story).lstrip()
                story = re.sub(r"[-()\"#/@:<>{}=~|]", "", story)
                output = st.text_area("Story", story, height=250)

                # text to speech
                API_URL = (
                    "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
                )
                headers = {
                    "Authorization": "Bearer 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"
                }
                payload = {"inputs": story}
                audio_bytes = requests.post(
                    API_URL, headers=headers, json=payload
                ).content
                st.audio(audio_bytes, format="audio/wav")

# help
elif choose == "Help":
    st.title("Help")
    with st.form("help"):
        col1, col2 = st.columns(2)

        with col1:
            st.text("\n")
            st.markdown("**Official email address:**")
            for i in range(2):
                st.text("\n")
            st.markdown("**Official tel:**")
        with col2:
            st.code("NLPITS_official2024@163.com", language="markdown")
            st.code("+44-7551167050", language="markdown")

        from_addr = "2387324762@qq.com"
        password = "pdewxqltfshtebia"
        to_addr = "2387324762@qq.com"
        smtp_server = "smtp.qq.com"
        message = st.text_area(
            label="",
            placeholder="Leave your problems or suggestions here...",
            label_visibility="collapsed",
        )
        contact = st.form_submit_button("Contact")
        if contact:
            msg = MIMEText(
                "Dear NLPITS Developer Team,\n" + "  " + message + "\n"
                "\n"
                "Best regards,\n"
                "Message from NLPITS",
                "plain",
                "utf-8",
            )
            msg["From"] = Header(from_addr)
            msg["To"] = Header(from_addr)
            subject = f"NLPITS: Help & Contact message"
            msg["Subject"] = Header(subject, "utf-8")

            try:
                smtpobj = smtplib.SMTP_SSL(smtp_server)
                smtpobj.connect(smtp_server, 465)
                smtpobj.login(from_addr, password)
                smtpobj.sendmail(from_addr, to_addr, msg.as_string())
                print("Send successfully")
            except smtplib.SMTPException:
                print("Fail to send")
            finally:
                smtpobj.quit()
            st.balloons()
            st.success("Your message is sent successfully.")
