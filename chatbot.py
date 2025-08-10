import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load PDF content
def extract_pdf_text(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load FAQs
def load_faqs(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# Prepare embeddings
def embed_texts(texts, model):
    return model.encode(texts)

# Find best match
def find_best_answer(question, pdf_embeddings, pdf_chunks, model, threshold=0.4):
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, pdf_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < threshold:
        return "Sorry, I couldn't find a relevant answer in my database."
    return pdf_chunks[best_idx]

# Main chatbot loop
def chatbot():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pdf_text = extract_pdf_text("document.pdf")
    faqs = load_faqs("faqs.txt")

    # Split PDF into chunks (optional: paragraphs, sentences)
    pdf_chunks = pdf_text.split('\n')  # or use smarter chunking
    pdf_embeddings = embed_texts(pdf_chunks, model)

    print("🤖 Ask me something (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        reply: str| None= find_best_answer(user_input, pdf_embeddings, pdf_chunks, model)
        print("Bot:", reply)
        

if __name__ == "__main__":
    chatbot()
