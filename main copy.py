import PyPDF2
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

generator = pipeline('text-generation', model='gpt2')
set_seed(42)

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


ebook_text = extract_text_from_pdf("CS-324 ML Topic 1 - Introduction to Machine Learning.pdf")


def split_into_chunks(text, chunk_size=256):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


chunks = split_into_chunks(ebook_text)


def get_embeddings(text):
    embeddings = model.encode(text)
    return embeddings

chunks_embeddings = [get_embeddings(chunk) for chunk in chunks]

def find_chunk(query, chunks, chunks_embeddings, threshold=0.3):
    embedded_query = get_embeddings(query)
    similarities = cosine_similarity([embedded_query], chunks_embeddings)
    max_similarity = np.max(similarities)
    if max_similarity < threshold:
        return None, max_similarity
    most_similar_chunk = np.argmax(similarities)
    return chunks[most_similar_chunk], max_similarity

user_query = input("Enter your query: ")
relevant_chunk, similarity_score = find_chunk(user_query, chunks, chunks_embeddings)



def generate_response(query, context):
    prompt = f"Ebook content:\n{context}\n\nUser query: {query}\n\nAnswer:"
    result_json = generator(prompt, max_length=1000, num_return_sequences=1, truncation=True)
    result =  result_json[0]['generated_text']
    answer = result.split("Answer:")[-1].strip()
    
    return answer



if relevant_chunk is None:
    print("Sorry, I couldn't find relevant content. Could you rephrase or ask about another topic?")
else:
    response = generate_response(user_query, relevant_chunk)
    print(response)