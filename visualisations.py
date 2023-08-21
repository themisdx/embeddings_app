import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from wordcloud import WordCloud
from collections import Counter
from stop_words import STOP_WORDS


def display_matplotlib_embeddings(model, focus_word, num_similar_words):
    fig, ax = plt.subplots(figsize=(15, 15))
    
    words = list(model.wv.index_to_key)
    vectors = [model.wv[word] for word in words]
    
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)

    focus_vector = model.wv[focus_word].reshape(1, -1)
    similarities = cosine_similarity(focus_vector, vectors)[0]
    
    # Filter out stop words and get top-N similar words
    sorted_indices = np.argsort(similarities)
    top_indices = [i for i in reversed(sorted_indices) if words[i].lower() not in STOP_WORDS][:num_similar_words]
    
    for index in top_indices:
        if words[index] == focus_word:
            ax.annotate(words[index], xy=(result[index, 0], result[index, 1]), xytext=(5, 2), 
                        textcoords='offset points', ha='right', va='bottom', color='red', weight='bold')
        else:
            ax.annotate(words[index], xy=(result[index, 0], result[index, 1]), xytext=(5, 2), 
                        textcoords='offset points', ha='right', va='bottom')
    
    ax.scatter(result[:, 0], result[:, 1], alpha=0.5)
    
    st.pyplot(fig)


def display_plotly_embeddings(model, focus_word, topn=150):
    # Get most similar words to the focus word
    similar_words = [word for word, _ in model.wv.most_similar(focus_word, topn=topn) if word.lower().strip(string.punctuation) not in STOP_WORDS]
    
    # Add the focus word to the list (if not a stop word)
    words = [focus_word] if focus_word.lower().strip(string.punctuation) not in STOP_WORDS else []
    words += similar_words
    
    vectors = [model.wv[word] for word in words]
    
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    
    # Prepare dataframe for plotly
    df = pd.DataFrame(result, columns=['x', 'y'])
    df['word'] = words
    df['color'] = ['red' if word == focus_word else 'blue' for word in words]
    
    fig = go.Figure(data=go.Scatter(x=df['x'], y=df['y'], mode='markers+text', 
                                    text=df['word'], marker_color=df['color'],
                                    marker_size=15, textposition='top center'))
    
    fig.update_layout(showlegend=False)
    
    st.plotly_chart(fig)


def display_wordcloud(texts):
    combined_text = ' '.join(texts)
    wordcloud = WordCloud(stopwords=STOP_WORDS, background_color="white", width=800, height=400).generate(combined_text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


def display_similarity_heatmap(model, focus_word, top_n=20):
    if focus_word in model.wv.index_to_key:
        words = list(model.wv.index_to_key)
        focus_vector = model.wv[focus_word].reshape(1, -1)
        similarities = cosine_similarity(focus_vector, model.wv.vectors)[0]
        
        # Filter out stop words and get top-N similar words
        sorted_indices = np.argsort(similarities)
        selected_indices = [i for i in reversed(sorted_indices) if words[i].lower() not in STOP_WORDS][:top_n]
        selected_words = [words[i] for i in selected_indices]
        heatmap_data = cosine_similarity(model.wv[selected_words])
        
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(heatmap_data, annot=True, xticklabels=selected_words, yticklabels=selected_words, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
import string


def display_word_frequencies(texts, top_n=20):
    combined_text = ' '.join(texts).split()
    
    # Process words: make lowercase and remove punctuation
    processed_words = [word.lower().strip(string.punctuation) for word in combined_text]
    counter = Counter(processed_words)
    
    # Remove stopwords from the counter
    for word in STOP_WORDS:
        if word in counter:
            del counter[word]
            
    most_common = counter.most_common(top_n)
    
    words, counts = zip(*most_common)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh(words, counts, color='skyblue')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Words')
    ax.invert_yaxis()
    st.pyplot(fig)


def display_top_similar_words_table(model, focus_word, num_similar_words):
    words = list(model.wv.index_to_key)
    vectors = [model.wv[word] for word in words]
    focus_vector = model.wv[focus_word].reshape(1, -1)
    similarities = cosine_similarity(focus_vector, vectors)[0]
    
    # Filter out stop words and get top-N similar words
    sorted_indices = np.argsort(similarities)
    top_indices = [i for i in reversed(sorted_indices) if words[i].lower() not in STOP_WORDS][:num_similar_words]
    
    top_words = [words[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]
    top_vectors = [vectors[i] for i in top_indices]

    df_table = pd.DataFrame({
        'Word': top_words,
        'Cosine Similarity': top_similarities,
        'Word Vector': top_vectors
    })

    st.table(df_table)
