import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def display_single_url_check(url, prediction, probability, features):
    if prediction is not None:
        result = "Malicious" if prediction == 1 else "Safe"
        color = "red" if prediction == 1 else "green"
        st.markdown(
            f"<h3 style='color: {color};'>This URL is {result}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"Confidence: {probability:.2f}")

        if features is not None:
            st.subheader("Extracted Features:")
            features_df = pd.DataFrame([features])
            features_df["timestamp"] = pd.to_datetime(
                features_df["timestamp"], unit="s"
            )
            features_df["hour_of_day"] = features_df["hour_of_day"].astype(int)
            features_df["day_of_week"] = features_df["day_of_week"].astype(int)
            st.dataframe(features_df)


def display_batch_results(results_df):
    st.subheader("Batch Processing Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total URLs processed", len(results_df))
    with col2:
        st.metric(
            "Detected as malicious",
            len(results_df[results_df["prediction"] == "Malicious"]),
        )
    with col3:
        st.metric(
            "Detected as safe", len(results_df[results_df["prediction"] == "Safe"])
        )

    fig_pie, ax_pie = plt.subplots()
    results_df["prediction"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=ax_pie
    )
    ax_pie.set_title("Safe vs Malicious URL Ratio")
    st.pyplot(fig_pie)

    fig, ax = plt.subplots()
    ax.hist(results_df["probability"], bins=20, edgecolor="black")
    ax.set_title("Distribution of Malicious Probabilities")
    ax.set_xlabel("Probability of being malicious")
    ax.set_ylabel("Number of URLs")
    st.pyplot(fig)

    st.subheader("Detailed Results")
    st.dataframe(results_df.style.format({"probability": "{:.2f}"}))

    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="url_check_results.csv",
        mime="text/csv",
    )
