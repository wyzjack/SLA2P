import os
import openai
import numpy as np
import scipy.io
from sklearn.utils import shuffle

# Set up OpenAI API key
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load the Arrhythmia dataset
data = scipy.io.loadmat("./data/arrhythmia.mat")
x_data = data['X']  # 518 samples
y_data = ((data['y']).astype(np.int32)).reshape(-1)

# Convert numerical vectors to text
def vector_to_text(vector):
    return ' '.join(map(str, vector))

# Convert the numerical vectors to text
x_data_text = [vector_to_text(sample) for sample in x_data]

# Shuffle the data
x_data_text, y_data = shuffle(x_data_text, y_data, random_state=42)

# Function to get GPT-3 embeddings for a document
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

# List to store embeddings
embeddings = []

# Iterate over the data and get embeddings
for sample_text in x_data_text:
    embedding = get_gpt3_embedding(sample_text)
    if embedding is not None:
        embeddings.append(embedding)

# Convert list of embeddings to numpy array
embeddings_array = np.vstack(embeddings)

# Save the dataset with attributes "X" (raw data), "y" (labels), and "embeddings" in .mat format
scipy.io.savemat('arrhythmia_gpt3_embeddings.mat', {
    'X': np.array(x_data_text, dtype='object'),
    'y': y_data,
    'embeddings': embeddings_array
})

print("Dataset saved successfully.")
