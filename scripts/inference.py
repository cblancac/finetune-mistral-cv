cv_text = """Working Experience
Miguel Carlos
Blanco
Cacharrón
2023 –
Present
(Remote)Consultancy | Start-up
Senior AI Data Scientist at Turing Challenge
Creation of QnA and Chatbots systems using vector databases
(Pinecone, Elasticsearch or Azure Cognitive Search), RAG and LLM
models (GPT3.5 or GPT4). Fine-tuned of LLMs, like GPT-3.5 Turbo
or Llama, to improve the performance of GPT4, reducing prices up
to 20x lower for inference. Client: Repsol, Microsoft and Mediapro.
2022 – 2023
(Remote)Consultancy | Start-up
Data Scientist Specialist at Foqum Analytics
Multi-label text classification with FastAi or Hugging Face (using
longformers). NER using regex for entities with fixed patterns,
and Hugging Face, in another case. Use of Haystack to build QA
pipelines, storing the documents in Elasticsearch. Client: Lefebvre.
2020 – 2022
(Hybrid)Consultancy
Engineer NLP at NTT Data
Extract texts from different types of documents using AWS Textract.
Classify them by using Neural Networks, SVM, Naive Bayes or Decis-
sion Trees. Get the entities present in the documents (Regex, Spacy
and TextBlob). Use of Microsoft Azure (Repos, Pipelines). Use of
Jira as an agile methodology. Clients: Iberdrola Spain, Vivo Brazil
and Entel Chile.
2019 – 2020
(Presential)Consultancy
Data Scientist at Strategy Big Data
Computer Vision - Improving the quality of the text extracted using
free OCR like pytesseract. Bayesian Optimizers to choose the best
combinations of hyperparameters.
Natural Language Processing - Extract the entities required in the
documents. Client: Banco Santander.
Data Scientist Specialist
i April 11, 1992
ć Gines, Sevilla (Spain)
× +34 603 468 505
Ɵ carblacac7@gmail.com
~ Spanish
Social Network
]
a
Education
Study
LinkedIn
2018 – 2019
Github
Hugging Face
Languages
2013 – 2018
SpanishĪ Ī Ī Ī Ī
EnglishĪ Ī Ī Ī Ī
FrenchĪ Ī Ī Ī Ī
Skills
CSIC and UIMP
Master in Data Science
Focus: Machine Learning and Deep Learning with Python and R.
Techniques of NLP, CV, Web Scrapping and ETL. Use of relational
and non-relational databases.
Master Theses
Sentiment analysis on Twitter, applying different Deep Learning
state-of-the-art techniques like transformers (BERT) using Python.
UNED
Degree in Mathematics
Focus: Programming (R and C++) applying knowledge of mathemat-
ics and statistics.
Bachelor Theses
Linear models in high dimensional small sampled datasets: Applied
to real data on Colon Cancer using R.
Courses
Programming:
PythonĪ Ī Ī Ī Ī
Bash scriptingĪ Ī Ī Ī Ī
RĪ Ī Ī Ī Ī
MATLAB, C++Ī Ī Ī Ī Ī
Tools:
(Click the image)
Hugging FaceĪ Ī Ī Ī ĪCertifications
TensorFlow, PytorchĪ Ī Ī Ī ĪMicrosoft
LangChainĪ Ī Ī Ī ĪSpacy, TextBlobĪ Ī Ī Ī ĪOpenCVĪ Ī Ī Ī Ī
Microsoft Certified: Azure AI Engineer Associate
2023Miguel Carlos
Blanco
Cacharrón
Data Scientist Specialist
Books of interest
O’Reilly
Packt
Natural Language with Transformers. Building Language Applica-
tions with Hugging Face
Lewis Tunstall, Leandro von Werra & Thomas Wolf
2022
Mastering spaCy: An end-to-end practical guide to implementing
NLP applications using the Python ecosystem
Duygu Altinok
2021
Deep Learning for Coders with Fastai and Pytorch: AI Applications
Without a PhD
Howard, J. and Gugger, S.
2020
About MeO’Reilly
I’m a person who enjoy with the small
details of the life, learning new things,
trying different foods and visiting new
places. The sport, family and friends
make my life easier.Personal Projects
A Normal Week2021
Program
Sport
2020
Sleep
Family
2019
Cryptocurrency
Detection of cryptocurrencies that will be listed soon on some plat-
forms or exchanges. Extraction of their information of interest con-
suming some APIs and analyze the sentiment related to these coins
through Twitter and Reddit. Bot that send this info automatically to
a premium channel of Telegram.
Sports Bets
Extraction of match odds through several leagues, countries and
sports in different bookmakers around the world. Detect arbitrage
between them. Create a Web Service that works in Flask for show
this information. Storage of the info about the user with sqlite3.
Stock Market
Loads data of different companies from Yahoo Finance source (open,
high, low, close, adjclose and volume), as well as scaling, shuffling,
normalizing and splitting. Sentiment analysis about such compa-
nies (GoogleNews, NewsApi, Newscatcher, Stocktwits, Tiingo, Red-
dit, Twitter and Finviz). Train models (LSTM - Neural Networks)
choosing the hyperparameters using Bayesian Optimizers. Predict
prices for the next seven days.
Extra-Curricular Activities
Music
Mobility
Sports
Taking my first steps with the guitar and the piano
Driving license, B. Own car. Passionate about travel and nature.
Throughout my life I have practiced a lot of sports, I competed
in ping pong, basketball and football. Nowadays, I an receiving
lessons of paddle and during my free time I do swim or run."""

import openai
import json

# Cliente actualizado compatible con vLLM
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# El esquema que has definido
SCHEMA = json.dumps({
    "certifications": "",
    "contact_detail": {
        "age": "", "email": "", "home_city": "", "mobile": "", "name": ""
    },
    "education": [],
    "gender": "",
    "industry": "",
    "skills": [],
    "software_tools": [],
    "work_abroad": "",
    "work_experience": []
}, separators=(",", ":"))

# Prompt del sistema
SYSTEM_PROMPT = (
    "You are an API that extracts structured JSON from resumes.\n"
    "Return *only* valid JSON matching exactly this schema:\n"
    f"{SCHEMA}"
)

# Función actualizada para la extracción
def extract_cv(cv_text):
    response = client.chat.completions.create(
        model="/workspace/finetune-mistral-cv/checkpoints/mistral-cv-merged-final",  # ruta o nombre que usa vLLM
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cv_text}
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content

# Ejemplo de uso
if __name__ == "__main__":
    result = extract_cv(cv_text)
    print(result)
