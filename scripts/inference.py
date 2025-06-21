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
"""

import openai
import json
import time
import logging

# Configuración básica de logging con timestamps
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 1️⃣ Medir tiempo de inicialización del cliente
client_init_start = time.time()
logger.info("Initializing OpenAI client for vLLM...")
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
client_init_end = time.time()
logger.info(f"Client initialized in {client_init_end - client_init_start:.3f}s")

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

def extract_cv(cv_text):
    # 2️⃣ Antes de la llamada a la inferencia
    logger.info("Preparing to send chat completion request...")
    request_start = time.time()
    
    response = client.chat.completions.create(
        model="mistral-cv-merged-final",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cv_text}
        ],
        max_tokens=1024, #Dejar en 512
        temperature=0.0,
    )
    
    # 3️⃣ Después de recibir la respuesta
    request_end = time.time()
    logger.info(f"Chat completion request took {request_end - request_start:.3f}s")

    return response.choices[0].message.content

from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_cv_wrapper(cv_text):
    return extract_cv(cv_text)

if __name__ == "__main__":
    cv_list = [cv_text] * 10  # You can replace this with 5 different CVs

    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(extract_cv_wrapper, cv) for cv in cv_list]

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error during CV extraction: {e}")

    end = time.time()
    logger.info(f"All CVs processed in {end - start:.2f} seconds")

    for idx, result in enumerate(results):
        print(f"\n=== Result {idx + 1} ===\n{result}")