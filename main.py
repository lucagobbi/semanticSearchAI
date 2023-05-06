import psycopg2
import string
import re

from langchain.vectorstores import Pinecone
from nltk.corpus import stopwords
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

conn = psycopg2.connect(database="aicompanydatadb",
                        host="localhost",
                        user="postgres",
                        password="postgres",
                        port="5432")

pinecone.init(
    api_key="8d1f626d-7a11-4e72-ace3-d983f000e39b",
    environment="us-west4-gcp"
)

index_name = "aicompanydata"

embeddings = OpenAIEmbeddings(openai_api_key="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")

llm = OpenAI(temperature=0, openai_api_key="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")
chain = load_qa_chain(llm, chain_type="stuff")
index = pinecone.Index(index_name)

def get_companies():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM company where description is not null")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    companies = [dict(zip(columns, row)) for row in rows]
    return companies


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
        print('Inserting emdedding for  ' + company['name'] + ' into pinecone.\nID: ' + company['vat_number'] + '\nDescription: ' + company['description'] + '\nEmbedding: ' + str(embedded_desc))
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
    docs = docsearch.similarity_search(query, k=3, include_metadata=True)
    resp = chain.run(input_documents=docs, question=query)
    print(resp)



def fetch():
    fetch_response = index.fetch(ids=['00988040176'])
    print(fetch_response)

    # query_response = index.query(
    #     top_k=10,
    #     include_values=True,
    #     include_metadata=True,
    # )
    # Pinecone.from_texts(cleaned_descriptions, embeddings, index_name=index_name)
    # query = "Quali aziende operano nel settore dell'information technology?"
    # docs = docsearch.similarity_search(query, include_metadata=True)
    # response = chain.run(input_documents=docs, question=query)
    # print(response)

def update():
    index.update(
        id="01800170985",
        set_metadata={"company_name": "Lomopress S.r.l."},
    )


if __name__ == '__main__':
    query_pinecone()