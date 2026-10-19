
import streamlit as st
import pandas as pd
from textblob import TextBlob

st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Sentiment Analysis")
st.write(
    "Analyze reviews and feedback as Positive, Negative, "
    "or Neutral using natural language processing."
)

# --------------------------------------------------
# 1. DATASET: 15 SAMPLE REVIEWS
# --------------------------------------------------

sample_reviews = [
    ("The product is excellent and works perfectly.", "Positive"),
    ("I am very happy with the service.", "Positive"),
    ("The teacher explained everything clearly.", "Positive"),
    ("The application is easy to use.", "Positive"),
    ("Amazing experience, I highly recommend it.", "Positive"),
    ("The delivery was fast and the product is great.", "Positive"),
    ("I love the quality and the design.", "Positive"),
    ("The product is broken and completely useless.", "Negative"),
    ("I am disappointed with the poor service.", "Negative"),
    ("The application crashes all the time.", "Negative"),
    ("The delivery was late and the package was damaged.", "Negative"),
    ("The teacher did not explain the topic well.", "Negative"),
    ("The product stopped working after one day.", "Negative"),
    ("The item arrived yesterday.", "Neutral"),
    ("The package contains a phone and a charger.", "Neutral")
]

df = pd.DataFrame(
    sample_reviews,
    columns=["Review", "Expected Sentiment"]
)


# --------------------------------------------------
# 2. SENTIMENT CLASSIFICATION
# --------------------------------------------------

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, round(polarity, 3)


def classify_dataframe(data):
    results = data.copy()

    predictions = results["Review"].apply(analyze_sentiment)

    results["Predicted Sentiment"] = predictions.apply(
        lambda result: result[0]
    )

    results["Polarity"] = predictions.apply(
        lambda result: result[1]
    )

    return results


# --------------------------------------------------
# 3. DISPLAY DATASET AND CLASSIFICATION TABLE
# --------------------------------------------------

st.header("1. Sample Dataset")

st.write("The dataset contains 15 sample reviews.")

with st.expander("View original sample dataset"):
    st.dataframe(df, use_container_width=True)

results = classify_dataframe(df)

st.header("2. Sentiment Classification Results")

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Polarity ranges from -1 (negative) to +1 (positive). "
    "Values close to zero are classified as neutral."
)


# --------------------------------------------------
# 4. SUMMARY STATISTICS AND CHART
# --------------------------------------------------

st.header("3. Sentiment Summary")

category_order = ["Positive", "Negative", "Neutral"]

counts = (
    results["Predicted Sentiment"]
    .value_counts()
    .reindex(category_order, fill_value=0)
)

col1, col2, col3 = st.columns(3)

col1.metric("Positive Reviews", int(counts["Positive"]))
col2.metric("Negative Reviews", int(counts["Negative"]))
col3.metric("Neutral Reviews", int(counts["Neutral"]))

st.subheader("Sentiment Distribution Chart")

chart_data = counts.rename_axis("Sentiment").reset_index(
    name="Number of Reviews"
)

st.bar_chart(
    chart_data,
    x="Sentiment",
    y="Number of Reviews",
    color="#4A90E2"
)

st.write("Percentage distribution")

percentages = (counts / len(results) * 100).round(1)

for category in category_order:
    st.write(f"**{category}:** {percentages[category]}%")


# --------------------------------------------------
# 5. TEST WITH THREE NEW STATEMENTS
# --------------------------------------------------

st.header("4. Test New Statements")

st.write(
    "Enter three new statements to test the classifier. "
    "The expected labels are examples for verification."
)

test_statements = [
    {
        "Review": "I absolutely love this wonderful application.",
        "Expected": "Positive"
    },
    {
        "Review": "The service was terrible and disappointing.",
        "Expected": "Negative"
    },
    {
        "Review": "The meeting starts at ten o'clock.",
        "Expected": "Neutral"
    }
]

test_df = pd.DataFrame(test_statements)

if st.button("Run Three Test Cases", type="primary"):
    test_results = classify_dataframe(
        test_df.rename(columns={"Expected": "Expected Sentiment"})
    )

    test_results["Correct Prediction"] = (
        test_results["Predicted Sentiment"]
        == test_results["Expected Sentiment"]
    )

    st.subheader("Test Results")

    st.dataframe(
        test_results,
        use_container_width=True,
        hide_index=True
    )

    correct = int(test_results["Correct Prediction"].sum())
    total = len(test_results)

    st.metric(
        "Test Accuracy",
        f"{correct}/{total} ({correct / total * 100:.1f}%)"
    )

    if correct == total:
        st.success("All three test predictions matched the expected labels.")
    else:
        st.warning(
            "Some predictions differed from the expected labels. "
            "This illustrates the limitations of sentiment analysis."
        )

st.divider()

st.info(
    "Note: This application uses a lexicon-based NLP method, "
    "not a trained deep-learning model. Predictions may be inaccurate "
    "for sarcasm, context-dependent language, or ambiguous statements."
)
