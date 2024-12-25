import os
import time
import openai
import h5py
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle
from transformers import GPT2Tokenizer
from requests.exceptions import Timeout

# Load the 20 Newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = np.array(newsgroups_data.data, dtype='object')  # Ensure dtype is 'object' for variable-length strings
labels = np.array(newsgroups_data.target)

# Set up OpenAI API key
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the GPT-3 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
max_tokens = 1024

# Function to get GPT-3 embeddings for a document chunk
def get_gpt3_embedding(text, retries=5, backoff_factor=2):
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",  # Specify the model to use
                input=text
            )
            return response['data'][0]['embedding']
        except (openai.error.APIError, Timeout) as e:
            if attempt < retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"Request failed: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Request failed after {retries} attempts: {e}")
                return None

# Function to split text into chunks within the token limit
def split_into_chunks(text, max_tokens):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# Select 360 samples per class
selected_documents = []
selected_labels = []

for class_label in np.unique(labels):
    class_indices = np.where(labels == class_label)[0]
    selected_indices = np.random.choice(class_indices, 360, replace=False)
    selected_documents.extend(documents[selected_indices])
    selected_labels.extend(labels[selected_indices])

# Convert to numpy arrays
selected_documents = np.array(selected_documents, dtype='object')  # Ensure dtype is 'object' for variable-length strings
selected_labels = np.array(selected_labels)

# Shuffle the selected samples
selected_documents, selected_labels = shuffle(selected_documents, selected_labels, random_state=42)

# List to store embeddings
embeddings = []

# Iterate over the selected documents and get embeddings
for doc in selected_documents:
    doc_embeddings = []
    chunks = split_into_chunks(doc, max_tokens)
    for chunk in chunks:
        if chunk.strip():  # Ensure the chunk is not empty
            chunk_embedding = get_gpt3_embedding(chunk)
            if chunk_embedding:
                doc_embeddings.append(chunk_embedding)
    # Average the embeddings if there are multiple chunks
    if len(doc_embeddings) > 1:
        doc_embedding = np.mean(doc_embeddings, axis=0)
    elif len(doc_embeddings) == 1:
        doc_embedding = doc_embeddings[0]
    else:
        # Handle the case where there are no valid chunks
        doc_embedding = np.zeros((1536,))  # Assuming the embedding size is 1536 for consistency
    embeddings.append(doc_embedding)

# Convert list of embeddings to numpy array
embeddings_array = np.vstack(embeddings)

# Save the dataset with attributes "X" (raw data) and "y" (labels) in HDF5 format
with h5py.File('20news.data', 'w') as h5_file:
    # Create dataset for raw data
    dt = h5py.special_dtype(vlen=str)
    raw_data_dataset = h5_file.create_dataset('X', (len(selected_documents),), dtype=dt)
    raw_data_dataset[:] = selected_documents
    
    # Create dataset for labels
    labels_dataset = h5_file.create_dataset('y', data=selected_labels)

    # Save the embeddings as well
    embeddings_dataset = h5_file.create_dataset('embeddings', data=embeddings_array)

print("Dataset saved successfully.")
