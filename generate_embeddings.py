import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def generate_and_save_embeddings():
    """
    Generate embeddings for the entire dataset and save them to a file
    """
    # Define paths
    DATASET_PATH = Path("dataset") / "webmd_dataset.csv"
    EMBEDDINGS_PATH = Path("dataset") / "embeddings.pkl"
    
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Dataset loaded successfully with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return
    
    # Find reviews column automatically
    def find_reviews_column(df):
        """Find the reviews column automatically"""
        possible_names = ['reviews', 'review', 'text', 'comment', 'feedback', 'description']
        
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        # If not found, return the column with longest average text length
        text_lengths = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                text_lengths[col] = avg_length
        
        if text_lengths:
            return max(text_lengths, key=text_lengths.get)
        
        return None
    
    reviews_col = find_reviews_column(df)
    if reviews_col is None:
        print("Error: No suitable text column found in the dataset")
        return
    
    print(f"Using column '{reviews_col}' for text analysis")
    
    # Clean data
    print("Cleaning data...")
    df_clean = df[df[reviews_col].notna()].copy()
    df_clean = df_clean[df_clean[reviews_col].astype(str).str.strip() != '']
    print(f"After cleaning: {len(df_clean)} rows")
    
    # Load model
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully")
    
    # Generate embeddings
    print("Generating embeddings... This may take a while...")
    texts = df_clean[reviews_col].astype(str).tolist()
    
    # Generate embeddings in batches to handle large datasets
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Create embedding data structure
    embedding_data = {
        'embeddings': embeddings,
        'indices': df_clean.index.tolist(),  # Store original dataframe indices
        'reviews_column': reviews_col,
        'dataset_shape': df_clean.shape,
        'metadata': {
            'model_name': 'all-MiniLM-L6-v2',
            'embedding_dim': embeddings.shape[1],
            'total_samples': len(embeddings)
        }
    }
    
    # Save embeddings
    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embedding_data, f)
    
    print("Embeddings saved successfully!")
    print(f"Embedding file size: {EMBEDDINGS_PATH.stat().st_size / (1024*1024):.2f} MB")
    
    return embedding_data

if __name__ == "__main__":
    embedding_data = generate_and_save_embeddings()
    if embedding_data:
        print("\n=== Embedding Generation Complete ===")
        print(f"Total embeddings: {embedding_data['metadata']['total_samples']}")
        print(f"Embedding dimension: {embedding_data['metadata']['embedding_dim']}")
        print(f"Model used: {embedding_data['metadata']['model_name']}")
        print("\nYou can now run the Streamlit app with pre-generated embeddings!")