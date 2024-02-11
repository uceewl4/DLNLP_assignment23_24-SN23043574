# """
# Author: uceewl4 uceewl4@ucl.ac.uk
# Date: 2024-02-11 15:31:32
# LastEditors: uceewl4 uceewl4@ucl.ac.uk
# LastEditTime: 2024-02-11 16:22:57
# FilePath: /DLNLP_assignment23_24-SN23043574/B/tmp.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# """

# from transformers import pipeline
# import sentencepiece
# import sacremoses

# # question-answering
# # qa = pipeline(
# #     "question-answering",
# #     model="deepset/roberta-base-squad2",
# #     tokenizer="deepset/roberta-base-squad2",
# # )
# # output = qa(
# #     "how many questions are asked",
# #     "Vaccines save millions of lives each year. Yet there are more and more people who are unsure about getting them. Why is that? So far, studies have looked at issues related directly to vaccines. But we think it might be something else. Could it have to do with how people's minds work? To find out, we asked 356 people different questions about what and how they think, and what they believe. \
# # What did we discover? People who like to trust their feelings and believe in supernatural things are more likely to be against vaccines. But those who think carefully and have some scientific knowledge generally trust vaccines. It seems the way our minds work can affect what we think about vaccines. This is important to remember when we talk about the importance of vaccines.",
# # )
# # print(output)

# # machine translation
# pipe = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
# output = pipe("what's your name")
# print(output)
# pipe = pipeline("translation_zh_to_en", model="Helsinki-NLP/opus-mt-zh-en")
# output = pipe("我叫安安")
# print(output)

# # import googletrans
# # from googletrans import Translator

# # translator = googletrans.Translator()
# # translate = translator.translate("what is your name", dest="zh-cn")
# # print(translate.text)

# # text summarization
# # import torch
# # from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# # model = T5ForConditionalGeneration.from_pretrained("t5-small")
# # tokenizer = T5Tokenizer.from_pretrained("t5-small")
# # t5_input_text = (
# #     "summarize: "
# #     + "Machine learning addresses the question of how to build computers that improve automatically through experience. It is one of today's most rapidly growing technical fields, lying at the intersection of computer science and statistics, and at the core of artificial intelligence and data science. Recent progress in machine learning has been driven both by the development of new learning algorithms and theory and by the ongoing explosion in the availability of online data and low-cost computation. The adoption of data-intensive machine-learning methods can be found throughout science, technology and commerce, leading to more evidence-based decision-making across many walks of life, including health care, manufacturing, education, financial modeling, policing, and marketing."
# # )
# # tokenized_text = tokenizer.encode(t5_input_text, return_tensors="pt", max_length=1024)
# # summary_ids = model.generate(tokenized_text, min_length=30, max_length=512)
# # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# # print(summary)


# # # rcoreference resolution
# # import spacy
# # import spacy_experimental
# # import os

# # # python3 -m spacy download en_core_web_sm
# # os.system("python3 -m spacy download en_core_web_sm")
# # nlp = spacy.load("en_core_web_sm")
# # nlp_coref = spacy.load("en_coreference_web_trf")

# # # use replace_listeners for the coref components
# # nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
# # nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

# # # we won't copy over the span cleaner
# # nlp.add_pipe("coref", source=nlp_coref)
# # nlp.add_pipe("span_resolver", source=nlp_coref)

# # doc = nlp("A man drop his umbrella which is bought by his son.")
# # print(doc.spans)

# text and code generation
from transformers import pipeline  # First line

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")  # Second line
prompt = "Wirte a code for binary searching"  # Third line
res = generator(prompt, max_length=50, do_sample=True, temperature=0.9)  # Fourth line
print(res[0]["generated_text"])

# # intent classification
# classifier = pipeline("text-classification", model="Falconsai/intent_classification")
# text = "what's your name"
# result = classifier(text)
# print(result)


# # # part of speech tagging
# # import nltk

# # nltk.help.upenn_tagset("MD")
# # text = "wikipedia is a free online encyclopedia"
# # word = nltk.word_tokenize(text)
# # nltk.pos_tag(word)
# # nltk.help.brown_tagset()


# from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# print(model.max_seq_length)
# corpus = "Science can be described as a complex, self-organizing, \
# and evolving network of scholars, projects, papers, and ideas. \
# This representation has unveiled patterns characterizing the emergence \
# of new scientific fields through the study of collaboration networks and \
# the path of impactful discoveries through the study of citation networks. \
# Microscopic models have traced the dynamics of citation accumulation, \
# allowing us to predict the future impact of individual papers. \
# SciSci has revealed choices and trade-offs that scientists face as \
# they advance both their own careers and the scientific horizon. \
# For example, measurements indicate that scholars are risk-averse, \
# preferring to study topics related to their current expertise, \
# which constrains the potential of future discoveries. \
# Those willing to break this pattern engage in riskier careers but become \
# more likely to make major breakthroughs. Overall, the highest-impact science \
# is grounded in conventional combinations of prior work but features unusual \
# combinations. Last, as the locus of research is shifting into teams, \
# SciSci is increasingly focused on the impact of team research, \
# finding that small teams tend to disrupt science and technology with \
# new ideas drawing on older and less prevalent ones. \
# In contrast, large teams tend to develop recent, popular ideas, obtaining high, but often short-lived, impact.".split(
#     "."
# )
# print(corpus)
# corpus_embedding = model.encode(corpus, show_progress_bar=True)
# # corpus_embeddings.shape
# query = corpus
# query_embedding = model.encode(query)
# print(util.cos_sim(query_embedding, corpus_embedding).mean() * 100)

# result = util.semantic_search(query_embedding, corpus_embedding)[0]
# for item in result:
#     print(round(item["score"], 2), corpus[item["corpus_id"]])


# import langchain_community
# from transformers import pipeline
# from langchain import LLMChain, PromptTemplate
# from langchain import HuggingFaceHub
# import requests


# def img2text(url):
#     pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
#     text = pipe(url)[0]["generated_text"]
#     return text


# API_URL = (
#     "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
# )
# headers = {"Authorization": "Bearer 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"}


# def img2text(filename):
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers=headers, data=data)
#     return response.json()


# output = img2text("B/bicycle.jpg")
# print(img2text("B/bicycle.jpg"))

# repo_id = "tiiuae/falcon-7b-instruct"
# hf_token = "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"
# llm = langchain_community.llms.HuggingFaceHub(
#     huggingfacehub_api_token=hf_token,
#     repo_id=repo_id,
#     verbose=False,
#     model_kwargs={"temperature": 0.1, "max_new_tokens": 1500},
# )


# def generate_story(scenario, llm):
#     template = """youare a story teller. y
#     ou get a scenario as an input text, and generates a short story out of it.
#     Context: {scenario}
#     Story:
#     """
#     prompt = PromptTemplate(template=template, input_variables=["scenario"])
#     chain = LLMChain(prompt=prompt, llm=llm)
#     story = chain.predict(scenario=scenario)
#     return story


# scenario = "A man is walking in a dark street"
# print(generate_story(scenario, llm))

# import requests


# def text2speech(text):
#     API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
#     headers = {"Authorization": "Bearer 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"}
#     payload = {"inputs": text}
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.content


# img_file = "B/bicycle.jpg"
# scenario = img2text(img_file)
# story = generate_story(scenario=scenario, llm=llm)
# print(story)
# audio_bytes = text2speech(story)
# from IPython.display import Audio

# Audio(audio_bytes)
# with open("myfile.wav", mode="bx") as f:
#     f.write(audio_bytes)
