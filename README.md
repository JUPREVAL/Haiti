# Haiti
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import gradio as gr

# Haiti travel data
data = [
    "Visit Citadelle Laferrière in Cap-Haïtien for stunning views.",
    "Try griot, a traditional Haitian fried pork dish.",
    "The best time to visit Haiti is between November and March.",
    "Jacmel is known for its vibrant art scene and colonial architecture.",
    "Tap-taps are colorful shared taxis used for local transport.",
    "Go meet Julie's Mom in the beautiful city of Belvil.",
    "Go to SODO.",
    "Welcome to the beautiful island of Haiti."
]

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data)

# KNN index
nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)

# Chatbot function
def ask_bot(question):
    query_vec = model.encode([question])
    distances, indices = nbrs.kneighbors(query_vec)
    return data[indices[0][0]]

# Gradio interface
iface = gr.Interface(
    fn=ask_bot,
    inputs="text",
    outputs="text",
    title="Mrs Juju's Haiti Travel Chatbot",
    description="Ask me anything about traveling to Haiti!"
)

iface.launch()
