import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from PIL import Image
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from pathlib import Path
import pickle
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(page_title="Topic Modeling", layout="wide")

# Define paths using pathlib
FILE_PATH = Path("dataset") / "webmd_dataset.csv"
LOGO_PATH = Path("logo.png")
EMBEDDINGS_PATH = Path("dataset") / "embeddings.pkl"

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")



@st.cache_data
def load_embeddings():
    """Load pre-generated embeddings from file"""
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embedding_data = pickle.load(f)
        return embedding_data
    except FileNotFoundError:
        st.error(f"Embeddings file not found at {EMBEDDINGS_PATH}. Please run generate_embeddings.py first.")
        return None
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

def get_filtered_embeddings(df_filtered, embedding_data):
    """Get embeddings for filtered dataframe based on original indices"""
    if embedding_data is None:
        return None
    
    # Get the indices of filtered dataframe
    filtered_indices = df_filtered.index.tolist()
    
    # Find positions in original embedding data
    original_indices = embedding_data['indices']
    positions = []
    
    for idx in filtered_indices:
        if idx in original_indices:
            pos = original_indices.index(idx)
            positions.append(pos)
    
    if not positions:
        return None
    
    # Extract embeddings for filtered data
    filtered_embeddings = embedding_data['embeddings'][positions]
    return filtered_embeddings

def add_logo_to_sidebar(logo_path, width=None, height=None):
    """
    Add a logo to the sidebar with custom width and height.
    
    Parameters:
    logo_path (str): Path to the logo image file
    width (int, optional): Desired width of the logo in pixels
    height (int, optional): Desired height of the logo in pixels
    """
    try:
        # Load the image
        logo = Image.open(logo_path)
        
        # Resize the logo if width or height is specified
        if width and height:
            logo = logo.resize((width, height), Image.LANCZOS)
        elif width:  # Only width is specified, maintain aspect ratio
            aspect_ratio = logo.height / logo.width
            new_height = int(width * aspect_ratio)
            logo = logo.resize((width, new_height), Image.LANCZOS)
        elif height:  # Only height is specified, maintain aspect ratio
            aspect_ratio = logo.width / logo.height
            new_width = int(height * aspect_ratio)
            logo = logo.resize((new_width, height), Image.LANCZOS)
        
        # Display logo in sidebar
        st.sidebar.image(logo)
        
    except FileNotFoundError:
        st.sidebar.error(f"Logo file not found: {logo_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading logo: {str(e)}")

@st.cache_data
def find_reviews_column(df):
    """Find the reviews column automatically"""
    # Look for common review column names
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

def get_filtered_data(df, filters):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    for col, values in filters.items():
        if values and col in df.columns:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    return filtered_df

def show_topic_reviews(df, topic_name, text_col, n_reviews=5):
    """Show sample reviews for a specific topic"""
    topic_reviews = df[df['topic_name'] == topic_name][text_col].head(n_reviews)
    
    st.subheader(f"Sample Reviews for {topic_name}")
    for i, review in enumerate(topic_reviews, 1):
        st.write(f"**{i}.** {review}")

def get_topic_name(topic_model, topic_id):
    """Generate a meaningful topic name from the top keywords (2-4 words)"""
    if topic_id == -1:
        return "Outliers/Miscellaneous"
    
    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        # Get the top 2-4 most relevant words
        num_words = min(len(topic_words), 4)
        keywords = [word for word, _ in topic_words[:num_words]]
        return " ".join(keywords)
    return f"Topic {topic_id}"

def safe_dimensionality_reduction(embeddings, n_components=2, method='umap'):
    """
    Safely perform dimensionality reduction with fallback options
    """
    n_samples = embeddings.shape[0]
    
    # Check if we have enough samples for the requested components
    if n_samples < n_components + 1:
        st.warning(f"Dataset too small ({n_samples} samples) for {n_components}D visualization. Using PCA instead.")
        # Use PCA as fallback for small datasets
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=min(n_components, n_samples-1))
        return reducer.fit_transform(embeddings)
    
    try:
        if method == 'umap':
            # Adjust UMAP parameters for small datasets
            n_neighbors = min(15, max(2, n_samples // 3))  # Adaptive n_neighbors
            min_dist = 0.1 if n_samples > 50 else 0.01     # Smaller min_dist for small datasets
            
            reducer = umap.UMAP(
                n_components=n_components, 
                random_state=42,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='cosine'  # Often works better for text embeddings
            )
            return reducer.fit_transform(embeddings)
    except Exception as e:
        st.warning(f"UMAP failed ({str(e)}). Using PCA as fallback.")
        # Fallback to PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=min(n_components, n_samples-1))
        return reducer.fit_transform(embeddings)

def generate_topic_name_with_gpt(topic_id, topic_model, client):
    """Generate a short, human-friendly topic name using GPT from keywords only."""
    keywords = topic_model.get_topic(topic_id)
    if not keywords:
        return f"Topic {topic_id}"
    
    # Use only the top 5–6 keywords
    top_keywords = [kw for kw, _ in keywords[:6]]

    prompt = f"""
    A clustering algorithm grouped some patient reviews into a topic.

    Keywords: {', '.join(top_keywords)}

    Suggest a short, clear, human-friendly topic name (max 5 words). 
    Return only the name.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",   # or gpt-4o-mini
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=20
    )
    name = response.choices[0].message.content.strip()
    return " ".join(name.splitlines()).strip() or f"Topic {topic_id}"



def generate_topic_summary_gpt4(topic_name, topic_keywords, sample_reviews):
    """
    Generate a summary of a topic using OpenAI GPT-4
    """
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
        return "OpenAI API key not configured."
    
    try:
        openai.api_key = OPENAI_API_KEY
        client = OpenAI()

        # Prepare the prompt
        prompt = f"""
        Analyze the following topic cluster from medical reviews and provide a comprehensive summary:
        
        Topic Name: {topic_name}
        Keywords: {', '.join(topic_keywords)}
        
        Sample Reviews:
        {chr(10).join([f'{i+1}. {review}' for i, review in enumerate(sample_reviews)])}
        
        Please provide a concise yet comprehensive summary of this topic that includes:
        1. The main themes or concerns expressed in these reviews
        2. Common patterns or sentiments
        3. Any notable insights about patient experiences
        4. Potential implications for healthcare providers
        
        Keep the summary professional and focused on medical insights.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You aoe a medical data analyst specializing in summarizing patient review data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Create page navigation
def data_viewer_page(df_filtered):
    """Data Viewer Page"""
    st.subheader("Dataset Overview")
    st.dataframe(df_filtered, use_container_width=True)

def topic_analysis_page(df_filtered, reviews_col, filters, selected_drugs, embedding_data):
    """Topic Analysis Page"""
    st.header("Topic Analysis & Visualizations")
    
    # Check if embeddings are available
    if embedding_data is None:
        st.error("⚠️ Pre-generated embeddings not found. Please run generate_embeddings.py first to create embeddings.")
        st.info("Run the following command in your terminal: `python generate_embeddings.py`")
        return
        
    # Get filtered embeddings BEFORE checking dataset size
    filtered_embeddings = get_filtered_embeddings(df_filtered, embedding_data)

    # Double check both dataframe size and embeddings size
    dataset_size = len(df_filtered)
    embedding_size = 0 if filtered_embeddings is None else len(filtered_embeddings)

    if embedding_size < 10:
        st.error("⚠️ Dataset too small for topic analysis. Please select more data or reduce filters.")
        st.info(f"Current dataset size: {dataset_size} reviews | Embeddings available: {embedding_size}. Minimum required: 10.")
        return

    
    # Adaptive parameters based on dataset size
    if dataset_size < 50:
        n_topics = min(3, dataset_size // 5)  # Fewer topics for small datasets
        min_topic_size = max(2, dataset_size // 10)  # Smaller minimum topic size
        st.info(f"Small dataset detected ({dataset_size} reviews). Using {n_topics} topics with min_topic_size={min_topic_size}")
    else:
        n_topics = 5
        min_topic_size = 10
    
    # Check if drugs are selected
    if not selected_drugs:
        st.warning("⚠️ Please select at least one drug from the filters in the sidebar before running topic analysis.")
        st.info("Use the 'Select Drugs' filter in the sidebar to choose drugs for analysis.")
        return
    
    # Show selected drugs info
    st.info(f"Topic analysis will be performed for: {', '.join(selected_drugs)}")
    
    # Check if filters have changed since last run
    filters_changed = False
    if 'previous_filters' in st.session_state:
        filters_changed = (st.session_state.previous_filters != str(filters))
    
    # Clear model if filters have changed
    if filters_changed and 'topic_model' in st.session_state:
        del st.session_state['topic_model']
        del st.session_state['topics']
        del st.session_state['df_with_topics']
        del st.session_state['topic_names']
        if 'topic_summaries' in st.session_state:
            del st.session_state['topic_summaries']
        st.info("Filters changed. Please run topic analysis again.")
    
    # Store current filters for comparison next time
    st.session_state.previous_filters = str(filters)
    
    # Updated button with custom color
    button_clicked = st.button("Run Topic Analysis", type="primary")
    
    # Apply custom CSS for button color
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #00FFFF !important;
        color: #000000 !important;
        border: 1px solid #00FFFF !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #00E6E6 !important;
        border: 1px solid #00E6E6 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if button_clicked:

        with st.spinner("Creating topic model..."):
            # Get text data
            texts = df_filtered[reviews_col].astype(str).tolist()
            
            # Get filtered embeddings
            filtered_embeddings = get_filtered_embeddings(df_filtered, embedding_data)
            
            if filtered_embeddings is None:
                st.error("Could not extract embeddings for filtered data. Please check your filters.")
                return
            
            if dataset_size < 50:
                min_df = 1
                max_df = 1.0  # don’t exclude common words for small sets
            else:
                min_df = max(1, dataset_size // 100)  # ~1% of docs
                max_df = 0.95

            vectorizer_model = CountVectorizer(
                stop_words="english",
                min_df=min_df,
                max_df=max_df,
                ngram_range=(1, 2)
            )

            topic_model = BERTopic(
                embedding_model=None,
                vectorizer_model=vectorizer_model,
                nr_topics=n_topics,
                min_topic_size=min_topic_size,
                verbose=False
            )
            
            # Fit the model using pre-computed embeddings
            topics, probs = topic_model.fit_transform(texts, filtered_embeddings)
            
            # Add topics to dataframe
            df_with_topics = df_filtered.copy()
            df_with_topics['topic'] = topics
            
            # Create meaningful topic names
            # === Create GPT-based topic names (cached in st.session_state) ===
            if 'topic_names' not in st.session_state:
                st.session_state['topic_names'] = {}

            topic_names = st.session_state['topic_names']

            client = None
            if OPENAI_API_KEY:
                try:
                    openai.api_key = OPENAI_API_KEY
                    client = OpenAI()
                except Exception as e:
                    st.warning(f"OpenAI client init failed, falling back to keyword names: {e}")

            unique_topic_ids = sorted(df_with_topics['topic'].unique().tolist())

            for topic_id in unique_topic_ids:
                if topic_id == -1:
                    topic_names[topic_id] = "Outliers/Miscellaneous"
                    continue

                if topic_id in topic_names and topic_names[topic_id]:
                    continue  # already cached

                if client:
                    try:
                        topic_names[topic_id] = generate_topic_name_with_gpt(topic_id, topic_model, client)
                    except Exception as e:
                        st.warning(f"GPT naming failed for topic {topic_id}: {e}")
                        topic_names[topic_id] = get_topic_name(topic_model, topic_id)
                else:
                    topic_names[topic_id] = get_topic_name(topic_model, topic_id)

            st.session_state['topic_names'] = topic_names
            df_with_topics['topic_name'] = df_with_topics['topic'].map(topic_names)

            
            # Add topic names to dataframe
            df_with_topics['topic_name'] = df_with_topics['topic'].map(topic_names)
            
            # Store in session state for use in visualizations
            st.session_state['topic_model'] = topic_model
            st.session_state['topics'] = topics
            st.session_state['df_with_topics'] = df_with_topics
            st.session_state['topic_names'] = topic_names
            st.session_state['filtered_embeddings'] = filtered_embeddings
        
        st.success("Topic modeling completed!")
    
    # Check if topic model exists
    if 'topic_model' not in st.session_state:
        st.warning("Please run topic analysis first using the button above.")
        return
    
    topic_model = st.session_state['topic_model']
    topics = st.session_state['topics']
    df_with_topics = st.session_state['df_with_topics']
    topic_names = st.session_state['topic_names']
    filtered_embeddings = st.session_state.get('filtered_embeddings')
    
    # Show current filter info
    st.info(f"Showing results for {len(df_with_topics)} reviews with current filters")
    
    # Visualization controls
    st.subheader("Visualization Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        viz_type = st.selectbox("Visualization Type:", ["2D Plot", "3D Plot"])
    with col2:
        show_reviews = st.checkbox("Show reviews on click", value=True)
    
    # Create visualizations
    if viz_type == "2D Plot":
        try:
            # Use pre-computed embeddings for visualization
            if filtered_embeddings is None:
                st.error("Embeddings not available for visualization.")
                return
                
            # Use safe dimensionality reduction
            reduced_embeddings = safe_dimensionality_reduction(filtered_embeddings, n_components=2)
            
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'topic': topics,
                'topic_name': df_with_topics['topic_name'],
                'text': df_with_topics[reviews_col].str[:100] + "..."
            })
            
            # Remove outlier topic (-1) for cleaner visualization
            plot_df = plot_df[plot_df['topic'] != -1]
            
            if len(plot_df) == 0:
                st.warning("No valid topics found for visualization. All data points are outliers.")
                return
            
            # Create interactive plot
            fig = px.scatter(
                plot_df,
                x='x',
                y='y',
                color='topic_name',
                hover_data={'text': True, 'x': False, 'y': False},
                title="Topic Clusters (2D Visualization)",
                labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Topic'}
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            
            # Display plot
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
            
            # Show reviews when point is clicked
            if show_reviews and event and 'selection' in event and event['selection']['points']:
                selected_point = event['selection']['points'][0]
                if 'customdata' in selected_point:
                    # Find the topic of selected point
                    point_index = selected_point['pointIndex']
                    selected_topic_name = plot_df.iloc[point_index]['topic_name']
                    show_topic_reviews(df_with_topics, selected_topic_name, reviews_col, 5)
        
        except Exception as e:
            st.error(f"Error creating 2D visualization: {str(e)}")
            st.info("This might be due to the dataset being too small or sparse. Try selecting more data.")
    
    elif viz_type == "3D Plot":
        try:
            # Use pre-computed embeddings for visualization
            if filtered_embeddings is None:
                st.error("Embeddings not available for visualization.")
                return
                
            # Use safe dimensionality reduction
            reduced_embeddings = safe_dimensionality_reduction(filtered_embeddings, n_components=3)
            
            # Handle case where we get fewer components than requested
            if reduced_embeddings.shape[1] < 3:
                st.warning("Dataset too small for 3D visualization. Showing 2D instead.")
                # Pad with zeros for 3D plot
                if reduced_embeddings.shape[1] == 2:
                    z_component = np.zeros((reduced_embeddings.shape[0], 1))
                    reduced_embeddings = np.hstack([reduced_embeddings, z_component])
                else:
                    st.error("Unable to create 3D visualization with current dataset.")
                    return
            
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'z': reduced_embeddings[:, 2],
                'topic': topics,
                'topic_name': df_with_topics['topic_name'],
                'text': df_with_topics[reviews_col].str[:100] + "..."
            })
            
            # Remove outlier topic (-1)
            plot_df = plot_df[plot_df['topic'] != -1]
            
            if len(plot_df) == 0:
                st.warning("No valid topics found for visualization. All data points are outliers.")
                return
            
            # Create 3D plot
            fig = px.scatter_3d(
                plot_df,
                x='x',
                y='y',
                z='z',
                color='topic_name',
                hover_data={'text': True, 'x': False, 'y': False, 'z': False},
                title="Topic Clusters (3D Visualization)",
                labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3', 'color': 'Topic'}
            )
            
            # Display plot
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
            
            # Show reviews when point is clicked
            if show_reviews and event and 'selection' in event and event['selection']['points']:
                selected_point = event['selection']['points'][0]
                if 'customdata' in selected_point:
                    point_index = selected_point['pointIndex']
                    selected_topic_name = plot_df.iloc[point_index]['topic_name']
                    show_topic_reviews(df_with_topics, selected_topic_name, reviews_col, 5)
        
        except Exception as e:
            st.error(f"Error creating 3D visualization: {str(e)}")
            st.info("This might be due to the dataset being too small or sparse. Try selecting more data.")
    
    # Topic distribution charts
    st.subheader("Topic Distribution")
    
    # Filter out outlier topic for distribution
    valid_topics_df = df_with_topics[df_with_topics['topic'] != -1]
    
    if len(valid_topics_df) == 0:
        st.warning("No valid topics found. All data points are classified as outliers.")
        return
    
    topic_counts = valid_topics_df['topic_name'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig_bar = px.bar(
            x=topic_counts.index,
            y=topic_counts.values,
            title="Topic Distribution (Bar Chart)",
            labels={'x': 'Topic', 'y': 'Number of Reviews'}
        )
        fig_bar.update_layout(xaxis_tickangle=45)
        event_bar = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun")
        
        # Show reviews when bar is clicked
        if show_reviews and event_bar and 'selection' in event_bar and event_bar['selection']['points']:
            selected_topic_name = event_bar['selection']['points'][0]['x']
            show_topic_reviews(df_with_topics, selected_topic_name, reviews_col, 5)
    
    with col2:
        # Pie chart
        fig_pie = px.pie(
            values=topic_counts.values,
            names=topic_counts.index,
            title='Topic Distribution (Pie Chart)'
        )
        event_pie = st.plotly_chart(fig_pie, use_container_width=True, on_select="rerun")
        
        # Show reviews when pie slice is clicked
        if show_reviews and event_pie and 'selection' in event_pie and event_pie['selection']['points']:
            selected_topic_name = event_pie['selection']['points'][0]['label']
            show_topic_reviews(df_with_topics, selected_topic_name, reviews_col, 5)
    
    # Topic details with GPT-4 summaries
    st.subheader("Topic Details")
    
    # Get valid topics (excluding outliers)
    valid_topics = sorted([t for t in df_with_topics['topic'].unique() if t != -1])
    
    if len(topic_counts) > 0:
        # Let the user select a topic by name
        selected_topic_name = st.selectbox("Select topic to explore:", list(topic_counts.index))
        
        if selected_topic_name:
            # Get topic ID from name
            selected_topic_id = df_with_topics[df_with_topics['topic_name'] == selected_topic_name]['topic'].iloc[0]
            
            # Get topic words
            topic_words = topic_model.get_topic(selected_topic_id)
            
            if topic_words:
                st.write(f"**Keywords:** {', '.join([word for word, _ in topic_words[:10]])}")
            
            # Show reviews for selected topic
            topic_reviews = df_with_topics[df_with_topics['topic_name'] == selected_topic_name][reviews_col]
            st.write(f"**Number of reviews:** {len(topic_reviews)}")
            
            # Display sample reviews
            st.write("**Sample Reviews:**")
            for i, review in enumerate(topic_reviews.head(6), 1):
                st.write(f"**{i}.** {review}")
            # Generate GPT-4 summary
            if st.button("Generate Topic Summary", key=f"summary_{selected_topic_id}"):
                with st.spinner("Generating AI summary with GPT-4..."):
                    # Extract keywords
                    topic_keywords = [word for word, _ in topic_words[:10]] if topic_words else []
                    
                    # Get sample reviews (first 10)
                    sample_reviews = topic_reviews.head(50).tolist()
                    
                    # Generate summary
                    summary = generate_topic_summary_gpt4(selected_topic_name, topic_keywords, sample_reviews)
                    
    
                    
                    # Store summary in session state to avoid regenerating
                    if 'topic_summaries' not in st.session_state:
                        st.session_state.topic_summaries = {}
                    st.session_state.topic_summaries[selected_topic_id] = summary
            
            # Check if we already have a summary for this topic
            if 'topic_summaries' in st.session_state and selected_topic_id in st.session_state.topic_summaries:
                st.subheader("AI-Generated Topic Summary")
                st.write(st.session_state.topic_summaries[selected_topic_id])

    # Generate summaries for all topics button
    if len(valid_topics) > 0:
        st.subheader("Generate All Topic Summaries")
        if st.button("Generate Summaries for All Topics"):
            if not OPENAI_API_KEY:
                st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
            else:
                import time
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if 'topic_summaries' not in st.session_state:
                    st.session_state.topic_summaries = {}
                
                for i, topic_id in enumerate(valid_topics):
                    if topic_id == -1:
                        continue
                        
                    topic_name = topic_names[topic_id]
                    status_text.text(f"Generating summary for {topic_name} ({i+1}/{len(valid_topics)})")
                    
                    # Get topic words
                    topic_words = topic_model.get_topic(topic_id)
                    
                    if topic_words:
                        # Extract keywords
                        topic_keywords = [word for word, _ in topic_words[:10]]
                        
                        # Get sample reviews
                        topic_reviews = df_with_topics[df_with_topics['topic'] == topic_id][reviews_col]
                        sample_reviews = topic_reviews.head(10).tolist()
                        
                        # Generate summary
                        summary = generate_topic_summary_gpt4(topic_name, topic_keywords, sample_reviews)
                        
                        # Store summary
                        st.session_state.topic_summaries[topic_id] = summary
                    
                    progress_bar.progress((i + 1) / len(valid_topics))
                    time.sleep(1)  # Avoid rate limiting
                
                status_text.text("All summaries generated!")
                st.success("All topic summaries have been generated successfully!")
    
    # Display all topic summaries if available
    if 'topic_summaries' in st.session_state and st.session_state.topic_summaries:
        st.subheader("All Topic Summaries")
        
        for topic_id, summary in st.session_state.topic_summaries.items():
            if topic_id == -1:
                continue
                
            topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
            with st.expander(f"Summary for {topic_name}"):
                st.write(summary)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(df_with_topics))
    with col2:
        valid_topics_count = len(valid_topics)
        st.metric("Topics Found", valid_topics_count)
    with col3:
        if valid_topics_count > 0:
            st.metric("Avg Reviews per Topic", f"{len(valid_topics_df)/valid_topics_count:.1f}")
        else:
            st.metric("Avg Reviews per Topic", "0")
    with col4:
        if len(topic_counts) > 0:
            st.metric("Largest Topic", f"{topic_counts.max()} reviews")
        else:
            st.metric("Largest Topic", "0 reviews")
    
    # Download results
    st.subheader("Download Results")
    
    # Prepare download data
    download_df = df_with_topics[[reviews_col, 'topic', 'topic_name']].copy()
    download_df.columns = ['review_text', 'topic_id', 'topic_name']
    
    csv = download_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="topic_modeling_results.csv",
        mime="text/csv"
    )

def main():
    # Add logo to sidebar
    add_logo_to_sidebar(LOGO_PATH, width=50)
    
    # Create navigation using sidebar
    st.sidebar.title("Sinta.ai")
    
    # Set default page
    if 'page' not in st.session_state:
        st.session_state.page = "Data Viewer"
    
    # Create navigation buttons with custom styling
    st.sidebar.markdown("""
    <style>
    .stButton button {
        background-color: #00FFFF !important;
        color: #000000 !important;
        border: 1px solid #00FFFF !important;
        width: 100% !important;
    }
    .stButton button:hover {
        background-color: #00E6E6 !important;
        border: 1px solid #00E6E6 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Data Viewer", use_container_width=True):
        st.session_state.page = "Data Viewer"
    
    if st.sidebar.button("Topic Analysis", use_container_width=True):
        st.session_state.page = "Topic Analysis"

    # Load embedding data once
    embedding_data = load_embeddings()

    try:
        # Load data from hardcoded path
        df = pd.read_csv(FILE_PATH)
        
        # Find reviews column
        reviews_col = find_reviews_column(df)
        
        if reviews_col is None:
            st.error("No text column found. Please select manually:")
            reviews_col = st.selectbox("Select reviews column:", df.columns)
        
        # Clean data
        df_clean = df[df[reviews_col].notna()].copy()
        df_clean = df_clean[df_clean[reviews_col].astype(str).str.strip() != '']
        
        # Sidebar Filters
        st.sidebar.subheader("Filters")
        filters = {}
        selected_drugs = []  # Initialize selected_drugs
        
        # Drug filter - Multi selector
        drug_cols = [col for col in ['Drug', 'drug', 'drugName', 'drug_name'] if col in df_clean.columns]
        if drug_cols:
            drug_col = drug_cols[0]
            # Remove empty/NaN values and get unique drugs
            unique_drugs = sorted([drug for drug in df_clean[drug_col].dropna().unique() if str(drug).strip() != ''])
            if len(unique_drugs) > 0:
                selected_drugs = st.sidebar.multiselect(
                    "Select Drugs: *Required for Topic Analysis*",
                    unique_drugs,
                    default=[],
                    help="You must select at least one drug to perform topic analysis"
                )
                if selected_drugs:
                    filters[drug_col] = selected_drugs
        
        # Age filter - Multi selector
        if 'Age' in df_clean.columns or 'age' in df_clean.columns:
            age_col = 'Age' if 'Age' in df_clean.columns else 'age'
            # Remove empty/NaN values and get unique ages
            unique_ages = sorted([age for age in df_clean[age_col].dropna().unique() if str(age).strip() != ''])
            if len(unique_ages) > 0:
                selected_ages = st.sidebar.multiselect(
                    "Select Ages:",
                    unique_ages,
                    default=[]
                )
                if selected_ages:
                    filters[age_col] = selected_ages
        
        # Satisfaction filter - Slider with range 1-5
        if 'Satisfaction' in df_clean.columns or 'satisfaction' in df_clean.columns:
            sat_col = 'Satisfaction' if 'Satisfaction' in df_clean.columns else 'satisfaction'
            
            # Convert satisfaction column to numeric, handling any non-numeric values
            df_clean[sat_col] = pd.to_numeric(df_clean[sat_col], errors='coerce')
            
            # Get the range of satisfaction values (should be 1-5)
            min_sat = df_clean[sat_col].min()
            max_sat = df_clean[sat_col].max()
            
            # Create slider for satisfaction range
            sat_range = st.sidebar.slider(
                "Select Satisfaction Range:",
                min_value=int(min_sat) if not pd.isna(min_sat) else 1,
                max_value=int(max_sat) if not pd.isna(max_sat) else 5,
                value=(int(min_sat) if not pd.isna(min_sat) else 1, int(max_sat) if not pd.isna(max_sat) else 5),
                step=1
            )
            
            # Filter based on the selected range
            filters[sat_col] = list(range(sat_range[0], sat_range[1] + 1))
        
        # Gender filter - Radio buttons
        gender_cols = [col for col in ['Gender', 'gender', 'Sex', 'sex'] if col in df_clean.columns]
        if gender_cols:
            gender_col = gender_cols[0]
            # Remove empty/NaN values and get unique genders
            unique_genders = sorted([gender for gender in df_clean[gender_col].dropna().unique() if str(gender).strip() != ''])
            if len(unique_genders) > 0:
                # Add "All" option to the beginning
                gender_options = ["All"] + list(unique_genders)
                selected_gender = st.sidebar.radio(
                    "Select Gender:",
                    gender_options
                )
                if selected_gender != "All":
                    filters[gender_col] = [selected_gender]
        
        # Apply filters
        if filters:
            df_filtered = get_filtered_data(df_clean, filters)
            st.sidebar.info(f"Filtered data: {len(df_filtered)} reviews")
        else:
            df_filtered = df_clean
            st.sidebar.info(f"Total data: {len(df_filtered)} reviews")
        
        # Display the appropriate page based on selection
        if st.session_state.page == "Data Viewer":
            data_viewer_page(df_filtered)
        elif st.session_state.page == "Topic Analysis":
            topic_analysis_page(df_filtered, reviews_col, filters, selected_drugs, embedding_data)

    except FileNotFoundError:
        st.error(f"File not found: {FILE_PATH}. Please make sure the file exists at the specified path.")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

if __name__ == "__main__":
    main()
