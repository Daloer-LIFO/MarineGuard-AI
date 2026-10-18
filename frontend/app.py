
import os

import pandas as pd
import requests
import streamlit as st
import folium
import plotly.express as px
from dotenv import load_dotenv
from streamlit_folium import st_folium

load_dotenv()

API_URL = os.getenv(
    "MARINEGUARD_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

st.set_page_config(
    page_title="MarineGuard AI",
    page_icon="⚓",
    layout="wide",
)

st.title("⚓ MarineGuard AI")
st.caption(
    "Maritime safety information and incident decision support"
)


def api_get(path):
    response = requests.get(
        f"{API_URL}{path}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def api_post(path, payload):
    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def show_api_error(exc):
    st.error(
        "Could not communicate with the backend. "
        "Check that FastAPI is running and the API URL is correct."
    )
    with st.expander("Technical details"):
        st.code(str(exc))


tab_chat, tab_incident, tab_dashboard = st.tabs([
    "AI Assistant",
    "Incident Management",
    "Dashboard",
])


with tab_chat:
    st.subheader("Maritime Knowledge Assistant")

    uploaded_file = st.file_uploader(
        "Upload a text-based maritime safety PDF",
        type=["pdf"],
    )

    if st.button("Upload and Index PDF"):
        if uploaded_file is None:
            st.warning("Please select a PDF first.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=120,
                )
                response.raise_for_status()

                st.success(
                    f"Document indexed. "
                    f"Chunks: {response.json()['chunks_indexed']}"
                )
            except requests.RequestException as exc:
                show_api_error(exc)

    question = st.text_area(
        "Ask a question about your uploaded documents",
        placeholder="What does the document say about maritime safety?",
    )

    if st.button("Ask MarineGuard AI"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                result = api_post(
                    "/ask",
                    {"question": question},
                )

                st.markdown("### Answer")
                st.write(result["answer"])

                st.markdown("### Retrieved sources")

                if result["sources"]:
                    st.dataframe(
                        pd.DataFrame(result["sources"]),
                        use_container_width=True,
                    )
                else:
                    st.info("No source passages were retrieved.")

            except requests.RequestException as exc:
                show_api_error(exc)


with tab_incident:
    st.subheader("Incident Classification and Recording")

    description = st.text_area(
        "Incident description",
        placeholder=(
            "Example: A cargo vessel reported an oil leak "
            "near the harbor."
        ),
    )

    location = st.text_input(
        "Location label (optional)",
        placeholder="Example: Sample coastal area",
    )

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=0.01,
        )

    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=0.0,
            step=0.01,
        )

    save_location = st.checkbox(
        "Save these coordinates",
        value=False,
    )

    if st.button("Classify and Save Incident"):
        if len(description.strip()) < 10:
            st.warning("Enter a more detailed incident description.")
        else:
            payload = {
                "description": description,
                "location": location or None,
                "latitude": latitude if save_location else None,
                "longitude": longitude if save_location else None,
            }

            try:
                result = api_post("/incidents", payload)

                st.success(
                    f"Incident saved. Predicted category: "
                    f"{result['category']}"
                )
                st.write(
                    "Model confidence:",
                    f"{result['confidence']:.1%}",
                )
                st.caption(result["note"])

            except requests.RequestException as exc:
                show_api_error(exc)


with tab_dashboard:
    st.subheader("Maritime Incident Dashboard")

    try:
        data = api_get("/incidents")
        incidents = data["incidents"]

        if not incidents:
            st.info(
                "No incidents recorded yet. "
                "Add a sample incident using the Incident Management tab."
            )
        else:
            df = pd.DataFrame(incidents)

            metric1, metric2 = st.columns(2)
            metric1.metric("Total incidents", len(df))
            metric2.metric(
                "Incident categories",
                df["category"].nunique(),
            )

            category_counts = (
                df["category"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Count")
            )

            fig = px.bar(
                category_counts,
                x="Category",
                y="Count",
                title="Incidents by Category",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Incident Records")
            st.dataframe(df, use_container_width=True)

            map_df = df.dropna(
                subset=["latitude", "longitude"]
            )

            map_df = map_df[
                map_df["latitude"].between(-90, 90)
                & map_df["longitude"].between(-180, 180)
            ]

            # Zero-zero is the form's default placeholder,
            # not a real sample incident location.
            map_df = map_df[
                ~(
                    (map_df["latitude"] == 0)
                    & (map_df["longitude"] == 0)
                )
            ]

            if not map_df.empty:
                st.markdown("### Incident Map")

                center = [
                    map_df["latitude"].mean(),
                    map_df["longitude"].mean(),
                ]

                maritime_map = folium.Map(
                    location=center,
                    zoom_start=5,
                )

                for _, row in map_df.iterrows():
                    folium.Marker(
                        location=[
                            row["latitude"],
                            row["longitude"],
                        ],
                        tooltip=row["category"],
                        popup=str(row["description"]),
                    ).add_to(maritime_map)

                st_folium(
                    maritime_map,
                    width=None,
                    height=450,
                    key="incident_map",
                )
            else:
                st.info(
                    "No valid coordinates are available for mapping."
                )

    except requests.RequestException as exc:
        show_api_error(exc)
