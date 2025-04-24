import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ----------- CACHED LOADERS ----------- #
@st.cache_data
def load_data():
    return (
        pd.read_csv("D:/project_4/cleaned_data.csv"),
        pd.read_csv("D:/project_4/pca_encoded_data.csv"),
    )

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ----------- RECOMMENDER CLASS ----------- #
class RestaurantRecommender:
    def __init__(self):
        self.cleaned_data, self.pca_data = load_data()
        self.cuisine_encoder = load_pickle("D:/project_4/cuisine_encoder.pkl")
        self.city_encoder = load_pickle("D:/project_4/city_encoder.pkl")
        self.pca_model = load_pickle("D:/project_4/pca_model.pkl")
        self.scaler = load_pickle("D:/project_4/scaler.pkl")
        self.kmeans = load_pickle("D:/project_4/kmeans_model.pkl")
        self.input_columns = load_pickle("D:/project_4/pca_input_columns.pkl")
        
        # Ensure cleaned_data and pca_data have matching indices
        self.cleaned_data = self.cleaned_data.reset_index(drop=True)
        self.pca_data = self.pca_data.reset_index(drop=True)

    def prepare_input(self, city, cuisine, rating, rating_count, cost):
        try:
            # One-hot encode city and cuisine
            city_enc = pd.DataFrame(self.city_encoder.transform([[city]]), 
                                 columns=self.city_encoder.get_feature_names_out())
            cuisine_enc = pd.DataFrame(self.cuisine_encoder.transform([[cuisine]]), 
                                     columns=self.cuisine_encoder.classes_)
            
            # Create numeric features DataFrame
            numeric = pd.DataFrame([{
                "rating": float(rating),
                "rating_count": float(rating_count),
                "cost": float(cost)
            }])
            
            # Combine all features
            full_input = pd.concat([numeric, city_enc, cuisine_enc], axis=1)
            
            # Ensure all expected columns are present, fill missing with 0
            for col in self.input_columns:
                if col not in full_input.columns:
                    full_input[col] = 0
            
            # Reorder columns to match training data
            full_input = full_input[self.input_columns]
            
            st.write("DEBUG: Final Input Vector", full_input.iloc[0:1])

            # Scale and transform
            scaled = self.scaler.transform(full_input)
            pca_input = self.pca_model.transform(scaled)
            return pca_input, city, cuisine
            
        except Exception as e:
            st.error(f"Error in input preparation: {str(e)}")
            return None, city, cuisine

    def recommend(self, pca_input, city, cuisine, method="Euclidean"):
        if pca_input is None:
            return None
            
        try:
            cluster = self.kmeans.predict(pca_input)[0]
            st.write(f"DEBUG: Predicted Cluster {cluster}")

            cluster_indices = np.where(self.kmeans.labels_ == cluster)[0]
            candidate_df = self.cleaned_data.iloc[cluster_indices].copy()

            st.write(f"DEBUG: Clustered Data Shape {candidate_df.shape}")

            # Filter by city and cuisine
            filtered_df = candidate_df[
                (candidate_df["city"].str.contains(city, case=False)) & 
                (candidate_df["cuisine"].str.contains(cuisine, case=False))
            ]

            if filtered_df.empty:
                st.warning("⚠️ No exact city/cuisine match found. Showing all cluster candidates.")
                filtered_df = candidate_df  # Fall back to all cluster candidates

            if filtered_df.empty:
                return None

            pca_candidates = self.pca_data.iloc[filtered_df.index]

            if method == "Euclidean":
                distances = np.linalg.norm(pca_candidates.values - pca_input, axis=1)
            else:
                distances = 1 - cosine_similarity(pca_candidates.values, pca_input).flatten()

            top_idx = np.argsort(distances)[:10]
            top_restaurants = filtered_df.iloc[top_idx].copy()
            top_restaurants["Distance"] = distances[top_idx]

            st.write("DEBUG: Final Recommendations", 
                     top_restaurants[["name", "city", "cuisine", "rating", "cost"]].head())

            return top_restaurants
            
        except Exception as e:
            st.error(f"Error in recommendation: {str(e)}")
            return None

# ----------- UI COMPONENTS ----------- #
def show_home():
    st.markdown("<h1 style='color:#FFA500;'>Welcome to the Smart Restaurant Recommender!</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00CED1;'>Your personalized restaurant guide based on city, cuisine, and preferences.</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#FF1493;'>Get the best recommendations tailored to your taste.</p>", unsafe_allow_html=True)

def show_recommendation_ui(recommender: RestaurantRecommender):
    st.markdown("<h1 style='color:#FFA500;'>🍽️ Smart Swiggy Restaurant Recommender</h1>", unsafe_allow_html=True)

    # --- Sidebar Filters --- 
    st.sidebar.header("🔧 Filter Options")

    # Get unique cities and cuisines
    cities = sorted(recommender.cleaned_data["city"].unique())
    city = st.sidebar.selectbox("Select City", cities)
    
    # Get cuisines available in selected city
    cuisine_options = sorted(recommender.cleaned_data[recommender.cleaned_data["city"] == city]["cuisine"].unique())
    cuisine = st.sidebar.selectbox("Select Cuisine", cuisine_options)

    # Get filtered data for min/max values
    filtered = recommender.cleaned_data[
        (recommender.cleaned_data["city"] == city) & 
        (recommender.cleaned_data["cuisine"] == cuisine)
    ]
    
    # Handle case when filtered is empty (use all restaurants in city)
    if filtered.empty:
        filtered = recommender.cleaned_data[
            (recommender.cleaned_data["city"] == city)
        ]

    # Rating slider with safe defaults
    min_rating = float(filtered["rating"].min())
    max_rating = float(filtered["rating"].max())
    
    # Ensure min and max are different
    if min_rating == max_rating:
        if min_rating > 0:
            min_rating = max_rating - 1.0  # Adjust min down by 1.0
        else:
            max_rating = min_rating + 1.0  # Adjust max up by 1.0
    
    default_rating = (min_rating + max_rating) / 2
    rating = st.sidebar.slider(
        "Select Rating", 
        min_value=min_rating, 
        max_value=max_rating, 
        value=default_rating, 
        step=0.1
    )

    # Cost slider with safe defaults
    min_cost = int(filtered["cost"].min())
    max_cost = int(filtered["cost"].max())
    
    # Ensure min and max are different
    if min_cost == max_cost:
        if min_cost > 0:
            min_cost = max_cost - 100  # Adjust min down by 100
        else:
            max_cost = min_cost + 100  # Adjust max up by 100
    
    default_cost = (min_cost + max_cost) // 2
    cost = st.sidebar.slider(
        "Select Cost", 
        min_value=min_cost, 
        max_value=max_cost, 
        value=default_cost, 
        step=10
    )

    # Use median rating count from filtered data
    default_rating_count = float(filtered["rating_count"].median())

    method = st.sidebar.radio("Distance Method", ["Euclidean", "Cosine"])
    recommend_btn = st.sidebar.button("🔍 Get Recommendations")

    # --- Main Display --- 
    if recommend_btn:
        with st.spinner("Finding the best recommendations..."):
            input_pca, city, cuisine = recommender.prepare_input(
                city, cuisine, rating, default_rating_count, cost
            )
            
            if input_pca is not None:
                results = recommender.recommend(input_pca, city, cuisine, method)

                if results is None or results.empty:
                    st.warning("No similar restaurants found. Try adjusting your filters.")
                else:
                    st.success(f"🎯 Top {len(results)} Recommended Restaurants")
                    
                    for _, row in results.iterrows():
                        st.markdown(f"<h4 style='color:#FF4500'>🍴 {row['name']}</h4>", unsafe_allow_html=True)
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.metric("Rating", f"{row['rating']} ⭐")
                            st.metric("Cost", f"₹{row['cost']}")
                        with cols[1]:
                            st.markdown(f"📍 **Location:** {row['city']}")
                            st.markdown(f"🍽️ **Cuisine:** {row['cuisine']}")
                        
                        if pd.notna(row.get("address", None)):
                            maps_url = f"https://www.google.com/maps/search/{row['address'].replace(' ', '+')}"
                            st.markdown(f"📌 <a href='{maps_url}' target='_blank'><b>View on Google Maps</b></a>", unsafe_allow_html=True)
                        if pd.notna(row.get("link", None)):
                            st.markdown(f"🔗 <a href='{row['link']}' target='_blank'><b>Order Online</b></a>", unsafe_allow_html=True)
                        st.markdown("---")

# ----------- MAIN ----------- #
def main():
    st.set_page_config("Smart Restaurant Recommender", layout="wide")
    page = st.sidebar.radio("Navigation", ["Home", "Recommendations"])
    recommender = RestaurantRecommender()

    if page == "Home":
        show_home()
    else:
        show_recommendation_ui(recommender)

if __name__ == "__main__":
    main()