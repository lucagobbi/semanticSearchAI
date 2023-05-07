import psycopg2
import string
import re

from langchain.vectorstores import Pinecone
from nltk.corpus import stopwords
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

from apikey import openai_api_key, pinecone_api_key

conn = psycopg2.connect(database="aicompanydatadb",
                        host="localhost",
                        user="postgres",
                        password="postgres",
                        port="5432")

pinecone.init(
    api_key=pinecone_api_key,
    environment="us-west4-gcp"
)

index_name = "aicompanydata"

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")
index = pinecone.Index(index_name)


def get_companies():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM company where description is not null")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    companies = [dict(zip(columns, row)) for row in rows]
    return companies

def get_vats():
    cursor = conn.cursor()
    cursor.execute("SELECT c.vat_number FROM company c where description is not null")
    vat_numbers = cursor.fetchall()
    return [x[0] for x in vat_numbers]


def clean_description(description):
    stop_words = stopwords.words('italian')
    cleaned_description = [word for word in description.split() if word.lower() not in stop_words]
    cleaned_description = ' '.join(cleaned_description)
    cleaned_description = cleaned_description.translate(str.maketrans('', '', string.punctuation))
    cleaned_description = re.sub(r'[^a-zA-Z\s]', '', cleaned_description.lower())
    return cleaned_description


def save_to_pinecone():
    companies = get_companies()
    for company in companies:
        company['description'] = clean_description(company['description'])
        embedded_desc = embeddings.embed_query(company['description'])
        print('Inserting embedding for  ' + company['name'] + ' into pinecone.\nID: ' + company[
            'vat_number'] + '\nDescription: ' + company['description'] + '\nEmbedding: ' + str(embedded_desc))
        index.upsert(
            vectors=[(
                company['vat_number'],
                embedded_desc,
            )]
        )
        print('Company inserted successfully')


def query_pinecone():
    query = input("Query: ")
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    docs = docsearch.similarity_search(query)
    resp = chain.run(input_documents=docs, question=query)
    print(resp)

def query_pinecone_score():
    query = input("Query: ")
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    docs = docsearch.similarity_search_with_score(query, k=10)
    for doc in docs:
        print('\nCompany Description: ' + doc[0].page_content)
        print('Score: ' + str(doc[1]))

def fetch():
    fetch_response = index.fetch(ids=['00988040176', '00295690176'])
    print(fetch_response)


def cluster():
    fetch_response = index.fetch(ids=get_vats())
    for item in fetch_response:
        print(item)

def update():
    companies = get_companies()
    for company in companies:
        index.update(
            id=company['vat_number'],
            set_metadata={"text": company['description']},
        )


if __name__ == '__main__':
    while True:
        choice = input("\n1) Query Pinecone without score 2) Query Pinecone with score \n")
        if int(choice) == 1:
            query_pinecone()
        elif int(choice) == 2:
            query_pinecone_score()
