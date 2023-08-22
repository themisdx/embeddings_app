import streamlit as st

from training import extract_text_from_pdfs, train_word2vec
from visualisations import display_plotly_embeddings, display_matplotlib_embeddings, display_wordcloud, display_similarity_heatmap, display_word_frequencies, display_top_similar_words_table

# Embeddings App
st.title('Word Embeddings Visualization App')

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    texts = extract_text_from_pdfs(uploaded_files)
    model = train_word2vec(texts)
    
    focus_word = st.text_input("Enter a focus word:", value="bullying").lower()
    num_similar_words = st.slider("Number of similar words to display:", 5, 50, 20)
    
    if st.button("Visualize"):
        st.write(f'Focusing on the word: {focus_word}')

        # Display the embeddings with matplotlib
        st.write("### Embeddings with Matplotlib")
        display_matplotlib_embeddings(model, focus_word, num_similar_words)

        # Display the embeddings
        st.write("### Embeddings with 150 most similar words")
        display_plotly_embeddings(model, focus_word)

        # Display the word cloud
        st.write("### Word Cloud")
        display_wordcloud(texts)

        # Display the similarity heatmap
        st.write("### Similarity Heatmap")
        display_similarity_heatmap(model, focus_word)

        # Display the histogram of word frequencies
        st.write("### Histogram of Word Frequencies")
        display_word_frequencies(texts)

        # Display table with top-N most similar words
        st.write("### Top Similar Words")
        display_top_similar_words_table(model, focus_word, num_similar_words)
