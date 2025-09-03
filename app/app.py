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
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Topic Modeling", layout="wide")

# Hardcoded file path - replace with your actual file path
FILE_PATH = "E:\\app\\POC\\dataset\\webmd_dataset.csv"  # Change this to your actual file path
LOGO_PATH = "E:\\app\\POC\\logo.png"  # Change this to your logo file path

@st.cache_resource
def load_model():
    """Load BERT model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

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

# Create page navigation
def data_viewer_page(df_filtered):
    """Data Viewer Page"""
    st.subheader("Dataset Overview")
    st.dataframe(df_filtered, use_container_width=True)

def topic_analysis_page(df_filtered, reviews_col, filters):
    """Topic Analysis Page"""
    st.header("Topic Analysis & Visualizations")
    
    # Use default topic modeling parameters
    n_topics = 5
    min_topic_size = 10
    
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
        st.info("Filters changed. Please run topic analysis again.")
    
    # Store current filters for comparison next time
    st.session_state.previous_filters = str(filters)
    
    if st.button("Run Topic Analysis", type="primary"):
        with st.spinner("Loading models..."):
            sentence_model = load_model()
        
        with st.spinner("Creating topic model..."):
            # Get text data
            texts = df_filtered[reviews_col].astype(str).tolist()
            
            # Create BERTopic model
            vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
            
            topic_model = BERTopic(
                embedding_model=sentence_model,
                vectorizer_model=vectorizer_model,
                nr_topics=n_topics,
                min_topic_size=min_topic_size,
                verbose=False
            )
            
            # Fit the model
            topics, probs = topic_model.fit_transform(texts)
            
            # Add topics to dataframe
            df_with_topics = df_filtered.copy()
            df_with_topics['topic'] = topics
            
            # Create meaningful topic names
            topic_names = {}
            for topic_id in df_with_topics['topic'].unique():
                topic_names[topic_id] = get_topic_name(topic_model, topic_id)
            
            # Add topic names to dataframe
            df_with_topics['topic_name'] = df_with_topics['topic'].map(topic_names)
            
            # Store in session state for use in visualizations
            st.session_state['topic_model'] = topic_model
            st.session_state['topics'] = topics
            st.session_state['df_with_topics'] = df_with_topics
            st.session_state['topic_names'] = topic_names
        
        st.success("Topic modeling completed!")
    
    # Check if topic model exists
    if 'topic_model' not in st.session_state:
        st.warning("Please run topic analysis first using the button above.")
        return
    
    topic_model = st.session_state['topic_model']
    topics = st.session_state['topics']
    df_with_topics = st.session_state['df_with_topics']
    topic_names = st.session_state['topic_names']
    
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
        # Get embeddings for visualization
        embeddings = topic_model._extract_embeddings(df_with_topics[reviews_col].tolist())
        
        # Reduce dimensions
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
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
        
        # Create interactive plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='topic_name',
            hover_data={'text': True, 'x': False, 'y': False},
            title="Topic Clusters (2D Visualization)",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'color': 'Topic'}
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
    
    elif viz_type == "3D Plot":
        # Get embeddings for visualization
        embeddings = topic_model._extract_embeddings(df_with_topics[reviews_col].tolist())
        
        # Reduce dimensions
        reducer = umap.UMAP(n_components=3, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
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
        
        # Create 3D plot
        fig = px.scatter_3d(
            plot_df,
            x='x',
            y='y',
            z='z',
            color='topic_name',
            hover_data={'text': True, 'x': False, 'y': False, 'z': False},
            title="Topic Clusters (3D Visualization)",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3', 'color': 'Topic'}
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
    
    # Topic distribution charts
    st.subheader("Topic Distribution")
    
    # Filter out outlier topic for distribution
    valid_topics_df = df_with_topics[df_with_topics['topic'] != -1]
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
    
    # Topic details
    st.subheader("Topic Details")
    
    # Get valid topics (excluding outliers)
    valid_topics = sorted([t for t in df_with_topics['topic'].unique() if t != -1])
    
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
        for i, review in enumerate(topic_reviews.head(5), 1):
            st.write(f"**{i}.** {review}")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(df_with_topics))
    with col2:
        valid_topics_count = len(valid_topics)
        st.metric("Topics Found", valid_topics_count)
    with col3:
        st.metric("Avg Reviews per Topic", f"{len(valid_topics_df)/valid_topics_count:.1f}")
    with col4:
        st.metric("Largest Topic", f"{topic_counts.max()} reviews")
    
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
    
    # Create navigation buttons
    if st.sidebar.button("Data Viewer", use_container_width=True):
        st.session_state.page = "Data Viewer"
    
    if st.sidebar.button("Topic Analysis", use_container_width=True):
        st.session_state.page = "Topic Analysis"

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
        # Drug filter - Multi selector
        drug_cols = [col for col in ['Drug', 'drug', 'drugName', 'drug_name'] if col in df_clean.columns]
        if drug_cols:
            drug_col = drug_cols[0]
            # Remove empty/NaN values and get unique drugs
            unique_drugs = sorted([drug for drug in df_clean[drug_col].dropna().unique() if str(drug).strip() != ''])
            if len(unique_drugs) > 0:
                selected_drugs = st.sidebar.multiselect(
                    "Select Drugs:",
                    unique_drugs,
                    default=[]
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
            topic_analysis_page(df_filtered, reviews_col, filters)

    except FileNotFoundError:
        st.error(f"File not found: {FILE_PATH}. Please make sure the file exists at the specified path.")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

if __name__ == "__main__":
    main()