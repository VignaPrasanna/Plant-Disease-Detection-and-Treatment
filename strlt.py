import streamlit as st
import torch
import torch.nn as nn
from groq import Groq
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "resnet50_plant_model.pth"
CLASSES_PATH = "classes.txt"
LLM_MODEL_ID = "openai/gpt-oss-120b"

GROQ_API_KEY = "enteryourgrokapikeyhere"  # Replace with your actual Groq API key
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes():
    with open(CLASSES_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(class_names)),
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model, class_names

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image: Image.Image, model, class_names, top_k=3):
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(class_names)))
    return [(class_names[i], p.item()) for p, i in zip(top_probs, top_idx)]


def parse_label(raw_label: str):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""
    if "___" in raw_label:
        plant, condition = raw_label.split("___", 1)
    else:
        plant, condition = "Unknown", raw_label
    clean = lambda s: s.replace("_", " ").replace(",", ", ").strip()
    return clean(plant), clean(condition)


# LLM call via Groq
def get_disease_info(plant: str, condition: str, api_key: str, model_id: str) -> str:
    client = Groq(api_key=api_key)

    user_prompt = (
        f"Plant: {plant}\nDisease: {condition}\n\n"
        "Give a response in markdown with exactly two sections:\n"
        "### Description\nA short description (2-3 sentences) covering the cause and symptoms.\n"
        "### Treatment\n4-6 concise bullet points covering cultural, organic, and chemical "
        "treatment/prevention methods.\n"
        "Keep the whole response under 200 words."
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You are an expert plant pathologist and agronomist. "
                           "Give accurate, practical, concise advice for farmers and gardeners.",
            },
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1500,  
        temperature=0.3,
        extra_body={"reasoning_effort": "low"},  # keeps replies fast and short
    )
    return response.choices[0].message.content

st.title("Plant Disease Detector")
st.caption("Upload a leaf image to identify the disease and get treatment advice.")

# Load model
try:
    model, class_names = load_model_and_classes()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Place it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analysing image..."):
        results = predict(image, model, class_names)

    top_label, top_conf = results[0]
    plant, condition = parse_label(top_label)

    col1, col2, col3 = st.columns(3)
    col1.metric("Plant", plant)
    col2.metric("Condition", condition)
    col3.metric("Confidence", f"{top_conf * 100:.1f}%")

    if condition.lower() == "healthy":
        st.success(f"This {plant} plant is healthy. No treatment needed.")
    else:
        with st.spinner("Asking the LLM for details..."):
            try:
                info = get_disease_info(plant, condition, GROQ_API_KEY, LLM_MODEL_ID)
                st.markdown(info)
            except Exception as e:
                st.error(f"LLM request failed: {e}")