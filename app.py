import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import cv2
from moviepy.editor import VideoFileClip
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, IntegerType

# Optional Kafka
try:
    KAFKA_ENABLED = True
except:
    KAFKA_ENABLED = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="VIRALQ", layout="wide")

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOAD FILES ----------------
model = pickle.load(open("model.pkl", "rb"))
tfidf_title = pickle.load(open("tfidf_title.pkl", "rb"))
tfidf_tags = pickle.load(open("tfidf_tags.pkl", "rb"))

# ---------------- SPARK ----------------
spark = SparkSession.builder.appName("ViralQ").getOrCreate()

spark_df = spark.read.csv("youtube_big_dataset.csv", header=True, inferSchema=True)

# ---------------- KEYWORDS ----------------
df_tags = spark_df.select("tags").toPandas()
df_tags['tags'] = df_tags['tags'].fillna("").astype(str)

all_tags = " ".join(df_tags['tags'].tolist())
words = [w.strip().lower() for w in all_tags.split(',') if w]
top_keywords = [w for w, _ in Counter(words).most_common(50)]

# ---------------- KAFKA STREAM ----------------
def get_kafka_data():
    try:
        schema = StructType() \
            .add("title", StringType()) \
            .add("views", IntegerType()) \
            .add("category", StringType())

        kafka_df = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "youtube_topic") \
            .load()

        value_df = kafka_df.selectExpr("CAST(value AS STRING)")

        json_df = value_df.select(
            from_json(col("value"), schema).alias("data")
        ).select("data.*")

        return json_df
    except:
        return None

# ---------------- HELPERS ----------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', str(text)).lower()

def trend_score(title, tags):
    text = title + " " + tags
    return sum(1 for w in top_keywords if w in text)

def hook_score(text):
    score = 0
    if "?" in text:
        score += 1
    if any(w in text for w in ["secret", "amazing", "shocking"]):
        score += 1
    if len(text) < 100:
        score += 1
    return score

def get_duration(video_path):
    clip = VideoFileClip(video_path)
    return clip.duration

def get_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else np.zeros((100,100,3))

def brightness(img):
    return np.mean(img)

# ---------------- LANDING ----------------
if st.session_state.page == "landing":

    st.markdown("<h1 style='text-align:center;margin-top:200px;'>🚀 VIRALQ</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>AI Video Virality System</h3>", unsafe_allow_html=True)

    if st.button("Enter Platform"):
        st.session_state.page = "menu"

# ---------------- MENU ----------------
elif st.session_state.page == "menu":

    st.title("📊 VIRALQ Dashboard")

    c1, c2, c3 = st.columns(3)

    if c1.button("🔥 Trending"):
        st.session_state.page = "trending"

    if c2.button("🎥 Predict"):
        st.session_state.page = "predict"

    if c3.button("📊 Compare"):
        st.session_state.page = "compare"

    st.success("⚡ Powered by Apache Spark")

# ---------------- TRENDING ----------------
elif st.session_state.page == "trending":

    st.title("🔥 Trending Videos")

    if st.button("⬅ Back"):
        st.session_state.page = "menu"

    use_kafka = st.toggle("Use Live Kafka Data")

    if use_kafka:
        kafka_data = get_kafka_data()

        if kafka_data:
            st.success("📡 Live Streaming Enabled")

            query = kafka_data.writeStream \
                .outputMode("append") \
                .format("memory") \
                .queryName("live_table") \
                .start()

            live_df = spark.sql("SELECT * FROM live_table").toPandas()
            st.dataframe(live_df)

        else:
            st.warning("Kafka not running. Showing static data.")

    categories = [row['category_id'] for row in spark_df.select("category_id").distinct().collect()]
    selected_cat = st.selectbox("Category", categories)

    filtered = spark_df.filter(spark_df.category_id == selected_cat)
    top_videos = filtered.orderBy(filtered.views.desc()).limit(10).toPandas()

    st.bar_chart(top_videos.set_index('title')['views'])

# ---------------- PREDICT ----------------
elif st.session_state.page == "predict":

    st.title("🎥 Virality Predictor")

    if st.button("⬅ Back"):
        st.session_state.page = "menu"

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Title")
        tags = st.text_input("Tags (comma separated)")
        hour = st.slider("Upload Hour", 0, 23, 18)

    with col2:
        video_file = st.file_uploader("Upload Video", type=["mp4"])
        if video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

    if st.button("🚀 Predict"):

        if not title or not tags:
            st.warning("Enter title and tags")
            st.stop()

        title_clean = clean_text(title)
        tags_clean = clean_text(tags)

        title_vec = tfidf_title.transform([title_clean]).toarray()
        tag_vec = tfidf_tags.transform([tags_clean]).toarray()

        t_len = len(title_clean)
        t_count = len(tags.split(','))
        trend = trend_score(title_clean, tags_clean)
        hook = hook_score(title_clean)

        duration, bright = 0, 0

        if video_file:
            with open("temp.mp4","wb") as f:
                f.write(video_bytes)

            duration = get_duration("temp.mp4")
            thumb = get_thumbnail("temp.mp4")
            bright = brightness(thumb)
            st.image(thumb)

        features = np.hstack((title_vec, tag_vec,
                              [[t_len, t_count, hour, duration, bright, trend, hook]]))

        score = model.predict(features)[0]

        # ---------------- RESULTS ----------------
        st.subheader("📊 Results")
        st.metric("Virality Score", round(score,3))
        st.progress(int(score*100))

        # ---------------- INTERPRETATION ----------------
        if score > 0.8:
            st.success("🚀 Extremely High Viral Potential")
        elif score > 0.6:
            st.info("📈 Good Potential")
        elif score > 0.4:
            st.warning("⚠ Needs Improvement")
        else:
            st.error("❌ Low Viral Potential")

        # ---------------- AI SUGGESTIONS ----------------
        st.subheader("🧠 AI Suggestions")

        suggestions = []

        if t_len < 20:
            suggestions.append("Make title longer and more descriptive.")
        if t_len > 80:
            suggestions.append("Shorten your title for better engagement.")
        if hook < 2:
            suggestions.append("Add emotional or curiosity words like 'Amazing', 'Secret'.")
        if t_count < 5:
            suggestions.append("Use at least 5-10 tags.")
        if trend < 2:
            suggestions.append("Include trending keywords.")
        if duration and duration > 300:
            suggestions.append("Shorten video duration.")
        if bright < 50:
            suggestions.append("Increase thumbnail brightness.")

        for s in suggestions:
            st.write("•", s)

        if not suggestions:
            st.success("Perfect optimization!")

        st.session_state.history.append({"title": title, "score": float(score)})

# ---------------- COMPARE ----------------
elif st.session_state.page == "compare":

    st.title("📊 Compare History")

    if st.button("⬅ Back"):
        st.session_state.page = "menu"

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.bar_chart(df.set_index('title'))
    else:
        st.info("No data yet")