import PyPDF2
from gensim.models import Word2Vec


# Function to extract text from uploaded PDFs
def extract_text_from_pdfs(uploaded_files):
    texts = []
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            texts.append(page.extract_text())
    return texts


# Train Word2Vec model
def train_word2vec(texts):
    sentences = [text.split() for text in texts]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.train(sentences, total_examples=len(sentences), epochs=100)
    return model
