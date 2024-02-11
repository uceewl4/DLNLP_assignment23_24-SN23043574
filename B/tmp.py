from transformers import pipeline
import sentencepiece
import sacremoses

# # question-answering
# qa = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     tokenizer="deepset/roberta-base-squad2",
# )
# output = qa(
#     "how many questions are asked",
#     "Vaccines save millions of lives each year. Yet there are more and more people who are unsure about getting them. Why is that? So far, studies have looked at issues related directly to vaccines. But we think it might be something else. Could it have to do with how people's minds work? To find out, we asked 356 people different questions about what and how they think, and what they believe. \
# What did we discover? People who like to trust their feelings and believe in supernatural things are more likely to be against vaccines. But those who think carefully and have some scientific knowledge generally trust vaccines. It seems the way our minds work can affect what we think about vaccines. This is important to remember when we talk about the importance of vaccines.",
# )
# # print(output)

# # machine translation
# pipe = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
# output = pipe("what's your name")
# print(output)
# pipe = pipeline("translation_zh_to_en", model="Helsinki-NLP/opus-mt-zh-en")
# output = pipe("我叫安安")
# print(output)

# import googletrans
# from googletrans import Translator

# translator = googletrans.Translator()
# translate = translator.translate("what is your name", dest="zh-cn")
# print(translate.text)

# # text summarization
# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# model = T5ForConditionalGeneration.from_pretrained("t5-small")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# t5_input_text = (
#     "summarize: "
#     + "Machine learning addresses the question of how to build computers that improve automatically through experience. It is one of today's most rapidly growing technical fields, lying at the intersection of computer science and statistics, and at the core of artificial intelligence and data science. Recent progress in machine learning has been driven both by the development of new learning algorithms and theory and by the ongoing explosion in the availability of online data and low-cost computation. The adoption of data-intensive machine-learning methods can be found throughout science, technology and commerce, leading to more evidence-based decision-making across many walks of life, including health care, manufacturing, education, financial modeling, policing, and marketing."
# )
# tokenized_text = tokenizer.encode(t5_input_text, return_tensors="pt", max_length=1024)
# summary_ids = model.generate(tokenized_text, min_length=30, max_length=512)
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print(summary)


# import spacy
# import spacy_experimental
# import os

# # python3 -m spacy download en_core_web_sm
# os.system("python3 -m spacy download en_core_web_sm")
# nlp = spacy.load("en_core_web_sm")
# nlp_coref = spacy.load("en_coreference_web_trf")

# # use replace_listeners for the coref components
# nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
# nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

# # we won't copy over the span cleaner
# nlp.add_pipe("coref", source=nlp_coref)
# nlp.add_pipe("span_resolver", source=nlp_coref)

# doc = nlp("A man drop his umbrella which is bought by his son.")
# print(doc.spans)

# # text and code generation
# from transformers import pipeline # First line
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B') # Second line
# prompt = "Wirte a code for binary searching" # Third line
# res = generator(prompt, max_length=50, do_sample=True, temperature=0.9) # Fourth line
# print(res[0]['generated_text'])

# intent classification
classifier = pipeline("text-classification", model="Falconsai/intent_classification")
text = "what's your name"
result = classifier(text)
print(result)


# part of speech tagging
import nltk

nltk.help.upenn_tagset("MD")
text = "wikipedia is a free online encyclopedia"
word = nltk.word_tokenize(text)
nltk.pos_tag(word)
nltk.help.brown_tagset()
