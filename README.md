# Swiggy-s-Restaurant-Recommendation-System-using-Streamlit-


A clustering-based restaurant recommendation system that suggests dining options based on location, cuisine preferences, ratings, and cost.

## 📌 Project Overview

**Domain**: Recommendation Systems and Data Analytics  
**Skills Demonstrated**: Data Preprocessing • One-Hot Encoding • K-Means Clustering • Cosine Similarity • Streamlit Development • Python

## 🎯 Problem Statement

Build a recommendation system that suggests restaurants to users based on:
- City location
- Cuisine preferences
- Rating thresholds
- Cost range

The system utilizes clustering and similarity measures to generate personalized recommendations through an intuitive Streamlit interface.

## 🏆 Business Use Cases

1. **Personalized Discovery**: Help users find restaurants matching their preferences
2. **Enhanced UX**: Streamline decision-making with tailored suggestions
3. **Market Analysis**: Identify popular cuisine and location trends
4. **Operational Insights**: Enable data-driven business decisions

## 🛠️ Technical Implementation

### Data Pipeline
```mermaid
graph TD
    A[Raw Data] --> B[Data Cleaning]
    B --> C[Feature Encoding]
    C --> D[Clustering Model]
    D --> E[Recommendation Engine]
    E --> F[Streamlit Interface]
