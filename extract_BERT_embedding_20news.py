import os
import h5py
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertModel

# Load the 20 Newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = np.array(newsgroups_data.data, dtype='object')  # Ensure dtype is 'object' for variable-length strings
labels = np.array(newsgroups_data.target)

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for a document
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings from the [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding.flatten()

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
    embedding = get_bert_embedding(doc)
    embeddings.append(embedding)

# Convert list of embeddings to numpy array
embeddings_array = np.vstack(embeddings)

# Save the dataset with attributes "X" (raw data) and "y" (labels) in HDF5 format
with h5py.File('20news_bert.data', 'w') as h5_file:
    # Create dataset for raw data
    dt = h5py.special_dtype(vlen=str)
    raw_data_dataset = h5_file.create_dataset('X', data=selected_documents, dtype=dt)
    
    # Create dataset for labels
    labels_dataset = h5_file.create_dataset('y', data=selected_labels)

    # Save the embeddings as well
    embeddings_dataset = h5_file.create_dataset('embeddings', data=embeddings_array)

print("Dataset saved successfully.")
