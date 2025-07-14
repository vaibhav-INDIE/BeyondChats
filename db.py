import os
import faiss
import numpy as np
import json
import pickle
import argparse
from openai import OpenAI
from dotenv import load_dotenv

def create_vector_database(username: str, input_json_file: str):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("posts", []) + data.get("comments", [])
    if not items:
        print("No data found.")
        return

    texts = []
    metadata = []
    for item in items:
        content = f"{item.get('title', '')}\n\n{item.get('selftext', '')}{item.get('body', '')}".strip()
        url = item.get("permalink")
        if content and url:
            texts.append(content)
            metadata.append({"source_url": url, "original_content": content})

    print(f"Processing {len(texts)} items...")

    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

    vectors = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, f"{username}_reddit.faiss")

    with open(f"{username}_reddit_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Vector DB and metadata saved.")

def run_vector_db(username: str):
    input_json_file = f"{username}_data.json"
    create_vector_database(username, input_json_file)
