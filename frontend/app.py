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
    "http://127.0.0.1:8000",
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

# --------------------------------------------------
# Persistent session-state messages
# --------------------------------------------------

if "incident_message" not in st.session_state:
    st.session_state.incident_message = None

if "incident_result" not in st.session_state:
    st.session_state.incident_result = None


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


def display_incident_message():
    """Display the saved incident message across Streamlit reruns."""

    message = st.session_state.incident_message

    if message is None:
        return

    message_type, message_text = message

    if message_type == "success":
        st.success(message_text)
    elif message_type == "error":
        st.error(message_text)
    elif message_type == "warning":
        st.warning(message_text)
    elif message_type == "info":
        st.info(message_text)


# --------------------------------------------------
# Application tabs
# --------------------------------------------------

tab_chat, tab_incident, tab_dashboard = st.tabs([
    "AI Assistant",
    "Incident Management",
    "Dashboard",
])


# ==================================================
# TAB 1: AI ASSISTANT
# ==================================================

with tab_chat:
    st.subheader("Maritime Knowledge Assistant")

    uploaded_file = st.file_uploader(
        "Upload a text-based maritime safety PDF",
        type=["pdf"],
        key="maritime_pdf_upload",
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

                upload_result = response.json()

                st.success(
                    f"Document indexed. Chunks: "
                    f"{upload_result['chunks_indexed']}"
                )

            except requests.RequestException as exc:
                show_api_error(exc)
            except (KeyError, ValueError) as exc:
                st.error(
                    f"The upload response was unexpected: {exc}"
                )

    question = st.text_area(
        "Ask a question about your uploaded documents",
        placeholder=(
            "What does the document say about maritime safety?"
        ),
        key="maritime_question",
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
                st.write(result.get("answer", "No answer was returned."))

                st.markdown("### Retrieved sources")

                sources = result.get("sources", [])

                if sources:
                    st.dataframe(
                        pd.DataFrame(sources),
                        use_container_width=True,
                    )
                else:
                    st.info("No source passages were retrieved.")

            except requests.RequestException as exc:
                show_api_error(exc)


# ==================================================
# TAB 2: INCIDENT MANAGEMENT
# ==================================================

with tab_incident:
    st.subheader("Incident Classification and Recording")

    # Display any previous result/message before processing
    # another incident.
    display_incident_message()

    if st.session_state.incident_result is not None:
        saved_result = st.session_state.incident_result

        st.write(
            "Model confidence:",
            f"{saved_result['confidence']:.1%}",
        )
        st.caption(saved_result.get("note", ""))

    description = st.text_area(
        "Incident description",
        placeholder=(
            "Example: A cargo vessel reported an oil leak "
            "near the harbor."
        ),
        key="incident_description",
    )

    location = st.text_input(
        "Location label (optional)",
        placeholder="Example: Sample coastal area",
        key="incident_location",
    )

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude (optional)",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=0.000001,
            format="%.6f",
            key="incident_latitude",
        )

    with col2:
        longitude = st.number_input(
            "Longitude (optional)",
            min_value=-180.0,
            max_value=180.0,
            value=0.0,
            step=0.000001,
            format="%.6f",
            key="incident_longitude",
        )

    st.caption(
        "Enter the actual coordinates to map this incident. "
        "Leave both values at 0.0 if the location is unknown."
    )

    if st.button("Classify and Save Incident"):
        # Clear the previous result when a new submission begins.
        st.session_state.incident_message = None
        st.session_state.incident_result = None

        if len(description.strip()) < 10:
            st.session_state.incident_message = (
                "warning",
                "Enter a more detailed incident description.",
            )
        else:
            # Avoid storing default zero-zero coordinates as a location.
            has_coordinates = not (
                latitude == 0.0 and longitude == 0.0
            )

            payload = {
                "description": description.strip(),
                "location": location.strip() or None,
                "latitude": latitude if has_coordinates else None,
                "longitude": longitude if has_coordinates else None,
            }

            try:
                result = api_post("/incidents", payload)

                category = result.get(
                    "category",
                    "Unknown",
                )

                confidence = result.get(
                    "confidence",
                    0.0,
                )

                note = result.get("note", "")

                st.session_state.incident_result = {
                    "confidence": confidence,
                    "note": note,
                }

                st.session_state.incident_message = (
                    "success",
                    f"Incident saved successfully. "
                    f"Predicted category: {category}",
                )

            except requests.RequestException as exc:
                st.session_state.incident_message = (
                    "error",
                    "Could not save the incident. "
                    "Check the backend and try again. "
                    f"Details: {exc}",
                )

            except (KeyError, TypeError, ValueError) as exc:
                st.session_state.incident_message = (
                    "error",
                    f"The backend returned an unexpected response: {exc}",
                )

        # Rerun so the saved message is rendered immediately
        # and remains visible on subsequent reruns.
        st.rerun()


# ==================================================
# TAB 3: DASHBOARD
# ==================================================

with tab_dashboard:
    st.subheader("Maritime Incident Dashboard")

    try:
        data = api_get("/incidents")
        incidents = data.get("incidents", [])

        if not incidents:
            st.info(
                "No incidents recorded yet. "
                "Add a sample incident using the Incident Management tab."
            )

        else:
            df = pd.DataFrame(incidents)

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Total incidents",
                len(df),
            )

            metric2.metric(
                "Incident categories",
                (
                    df["category"].nunique()
                    if "category" in df.columns
                    else 0
                ),
            )

            if "category" in df.columns:
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

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.markdown("### Incident Records")
            st.dataframe(
                df,
                use_container_width=True,
            )

            # Ensure coordinate columns exist before mapping.
            if (
                "latitude" in df.columns
                and "longitude" in df.columns
            ):
                df["latitude"] = pd.to_numeric(
                    df["latitude"],
                    errors="coerce",
                )

                df["longitude"] = pd.to_numeric(
                    df["longitude"],
                    errors="coerce",
                )

                map_df = df.dropna(
                    subset=["latitude", "longitude"]
                )

                map_df = map_df[
                    map_df["latitude"].between(-90, 90)
                    & map_df["longitude"].between(-180, 180)
                ]

                # Exclude the default 0, 0 placeholder.
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
                        category = row.get(
                            "category",
                            "Unknown category",
                        )

                        incident_description = row.get(
                            "description",
                            "No description available",
                        )

                        folium.Marker(
                            location=[
                                row["latitude"],
                                row["longitude"],
                            ],
                            tooltip=str(category),
                            popup=str(incident_description),
                        ).add_to(maritime_map)

                    st_folium(
                        maritime_map,
                        width=None,
                        height=450,
                        key="incident_map",
                    )

                else:
                    st.info(
                        "No valid coordinates are available for mapping. "
                        "Enter real latitude and longitude values "
                        "when recording an incident."
                    )

            else:
                st.info(
                    "The incident records do not contain latitude "
                    "and longitude fields yet."
                )

    except requests.RequestException as exc:
        show_api_error(exc)
