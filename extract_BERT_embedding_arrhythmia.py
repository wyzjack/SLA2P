import os
import numpy as np
import scipy.io
import torch
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertModel

# Load the Arrhythmia dataset
data = scipy.io.loadmat("./data/arrhythmia.mat")
x_data = data['X']  # 518 samples
y_data = ((data['y']).astype(np.int32)).reshape(-1)

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Convert numerical vectors to text
def vector_to_text(vector):
    return ' '.join(map(str, vector))

# Convert the numerical vectors to text
x_data_text = [vector_to_text(sample) for sample in x_data]

# Shuffle the data
x_data_text, y_data = shuffle(x_data_text, y_data, random_state=42)

# Function to get BERT embeddings for a document
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings from the [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding.flatten()

# List to store embeddings
embeddings = []

# Iterate over the data and get embeddings
for sample_text in x_data_text:
    embedding = get_bert_embedding(sample_text)
    embeddings.append(embedding)

# Convert list of embeddings to numpy array
embeddings_array = np.vstack(embeddings)

# Save the dataset with attributes "X" (raw data), "y" (labels), and "embeddings" in .mat format
scipy.io.savemat('./arrhythmia_bert.mat', {
    'X': embeddings_array,
    'y': y_data
})

print("Dataset saved successfully.")
