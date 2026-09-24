"""Main Streamlit application for anime recommendations."""

import streamlit as st
from client import RecommenderClient


def main():
    """Run the Streamlit application."""
    st.set_page_config(page_title="Senpai Suggest", layout="wide")
    st.title("🎬 Senpai Suggest")
    st.markdown("Your personalized anime recommendation engine")

    # Initialize client
    client = RecommenderClient()

    # User input
    st.sidebar.header("Find Recommendations")
    user_id = st.sidebar.text_input(
        "Enter your user ID:", value="1", help="Your unique user identifier"
    )

    if st.sidebar.button("Get Recommendations"):
        try:
            with st.spinner("Fetching recommendations..."):
                recommendations = client.get_recommendations(user_id)

            st.success("Recommendations retrieved!")
            st.dataframe(recommendations)

            # Log the request
            st.session_state.last_query = {
                "user_id": user_id,
                "timestamp": st.datetime.now().isoformat(),
            }
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Backend is currently unavailable. Please try again later.")


if __name__ == "__main__":
    main()
