# Word Embeddings Visualization Tool

Visualize word embeddings from PDF documents. Dive deep into relationships, similarities, and clusterings of words from your textual content.

![Word Embedding Visualization](images/user_interface.png) 
<!-- Replace the above link with an actual screenshot or logo link -->

## 🌟 Features

- **Upload PDFs**: Seamlessly upload one or multiple PDF documents.
- **Dynamic Word Embedding**: Generates embeddings using the powerful Word2Vec algorithm.
- **Visualizations**: Dive into various visualizations:
  - 📊 **Plotly Scatter Plot**: Highlighting the focus word and its most similar words.
  - 📉 **Matplotlib Scatter Plot**: Distinct color and style for the focus word.
  - ☁️ **Word Cloud**: Visual representation of word frequency.
  - 📊 **Histogram of Word Frequencies**: Top 20 words spotlight.
  - 🌡️ **Similarity Heatmap**: Cosine similarities between the focus word and the document words.
  - 🌲 **Clustering of Words**: Hierarchical clustering for word groupings.
- **Customizable**: Choose your focus word and the number of similar words for tailored insights.

## 🚀 Getting Started

### Prerequisites

Ensure Python is installed along with necessary libraries like Streamlit, Gensim, Plotly, Matplotlib, etc.

### Installation & Running

1. Clone the repository:

    ```bash
   git clone <repository_url>
 ```

2. Navigate to the directory:

  ```bash
  cd path/to/directory
 ```


3. Run the Streamlit app:
  
  ```bash
  streamlit run app.py
 ```

## 🙌 Contribute
Feel the need for an extra feature or want to fix a bug? Open an issue or send us a pull request. Contributions are more than welcome!

## 📜 License
This project is under the MIT License.
