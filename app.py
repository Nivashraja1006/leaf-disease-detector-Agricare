import os
import io
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
from gtts import gTTS
from model_helpers import load_model_and_encoder, preprocess_image_for_model, demo_prediction

# Optional OpenAI import
try:
    import openai
except:
    openai = None

app = Flask(__name__)
CORS(app)

# -------------------------------
# Load ML Model
# -------------------------------
MODEL_PATH = "model.pkl"
ENCODER_PATH = "label_encoder.pkl"
model, label_encoder = load_model_and_encoder(MODEL_PATH, ENCODER_PATH)

# -------------------------------
# DISEASE DATABASE
# -------------------------------
DISEASE_DB = {
    "Tomato_Blight": {
        "crop": "Tomato",
        "disease_en": "Early Blight",
        "disease_ta": "முந்தைய பூஞ்சை (Early Blight)",
        "symptoms_en": "Dark spots on older leaves, yellowing from edge.",
        "symptoms_ta": "முந்தைய இலைகளில் கருப்பு துளைகள், ஓரங்களில் மஞ்சள்.",
        "treatment_en": "Remove affected leaves, apply copper fungicide, crop rotation.",
        "treatment_ta": "பாதிக்கப்பட்ட இலைகளை அகற்று, காப்பர் பூஞ்சை மருந்து பயன்படுத்து, பயிர் மாறுதல்.",
        "prevention_en": "Avoid overhead watering, improve air circulation.",
        "prevention_ta": "மேல்நீர் தெளித்தல் தவிர்க்கவும், காற்றோட்டம் மேம்படுத்து."
    },

    "Powdery_Mildew": {
        "crop": "General",
        "disease_en": "Powdery Mildew",
        "disease_ta": "பவுடரி பூஞ்சை",
        "symptoms_en": "White powder coating on leaves.",
        "symptoms_ta": "இலையில் வெள்ளை பொடி போன்ற மேல் பரப்பு.",
        "treatment_en": "Use sulfur fungicide, prune infected parts.",
        "treatment_ta": "சல்பர் மருந்து பயன்படுத்தவும், பாதிக்கப்பட்ட பகுதிகளை வெட்டவும்.",
        "prevention_en": "Provide airflow, avoid dense planting.",
        "prevention_ta": "அடர்த்தியான நடவு தவிர்க்கவும், காற்றோட்டம் ஏற்படுத்தவும்."
    },
    
"Pepper_Leaf_Spot": {
    "crop": "Pepper / Capsicum",
    "disease_en": "Leaf Spot",
    "disease_ta": "இலை புள்ளி நோய் (Leaf Spot)",
    "symptoms_en": "Small brown circular spots on leaves, yellow halo, leaves may drop early.",
    "symptoms_ta": "இலைகளில் சிறிய பழுப்பு வட்ட புள்ளிகள், மஞ்சள் வட்டம் சுற்றி காணப்படும், இலைகள் முற்றிலும் விழும் வாய்ப்பு உள்ளது.",
    "treatment_en": "Spray copper oxychloride or mancozeb, remove heavily infected leaves.",
    "treatment_ta": "காப்பர் ஆக்ஸிகுளோரைடு அல்லது மாங்கோசெப் மருந்து தெளிக்கவும், கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
    "prevention_en": "Avoid overhead watering, maintain proper spacing to improve airflow.",
    "prevention_ta": "மேல்நீர் தெளிப்பதை தவிர்க்கவும், காற்றோட்டம் மேம்பட தாவரங்களுக்கு இடைவெளி அளிக்கவும்."
},

"Pepper_Anthracnose": {
    "crop": "Pepper / Capsicum",
    "disease_en": "Anthracnose",
    "disease_ta": "ஆன்த்ரக்னோஸ் நோய்",
    "symptoms_en": "Dark sunken spots on fruits, orange fungal rings, fruit rotting.",
    "symptoms_ta": "காய்களில் கருப்பு வட்ட பள்ளங்கள், ஆரஞ்சு நிற பூஞ்சை வளையங்கள், காய் அழுகுதல்.",
    "treatment_en": "Apply carbendazim or azoxystrobin fungicides, avoid infected fruits.",
    "treatment_ta": "கார்பெண்டசிம் அல்லது அஜாக்ஸிஸ்ட்ரோபின் மருந்துகள் தெளிக்கவும், பாதிக்கப்பட்ட காய்களை அகற்றவும்.",
    "prevention_en": "Use disease-free seeds, avoid water stagnation, rotate crops.",
    "prevention_ta": "நோயற்ற விதைகள் பயன்படுத்தவும், நீர் தேக்கம் தவிர்க்கவும், பயிர் மாறுதல் செய்யவும்."
}

}

# -------------------------------
# ROUTES
# -------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect")
def detect_page():
    return render_template("detect.html")

# -------------------------------
# PREDICT API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream).convert("RGB")

    # Try actual model
    if model is not None:
        try:
            x = preprocess_image_for_model(image)
            pred = model.predict([x])

            if label_encoder:
                label = label_encoder.inverse_transform(pred)[0]
            else:
                label = str(pred[0])

            # confidence
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba([x])[0]))

            # database lookup
            info = DISEASE_DB.get(label, DISEASE_DB["Powdery_Mildew"])

            return jsonify({
                "crop": info["crop"],
                "disease": info["disease_en"],
                "disease_ta": info["disease_ta"],
                "symptoms_en": info["symptoms_en"],
                "symptoms_ta": info["symptoms_ta"],
                "treatment_en": info["treatment_en"],
                "treatment_ta": info["treatment_ta"],
                "prevention_en": info["prevention_en"],
                "prevention_ta": info["prevention_ta"],
                "confidence": confidence,
                "source": "model"
            })
        except Exception as e:
            print("Model error:", e)

    # fallback — demo mode
    demo = demo_prediction(image)
    info = DISEASE_DB.get(demo["label"], DISEASE_DB["Powdery_Mildew"])

    return jsonify({
        "crop": info["crop"],
        "disease": info["disease_en"],
        "disease_ta": info["disease_ta"],
        "symptoms_en": info["symptoms_en"],
        "symptoms_ta": info["symptoms_ta"],
        "treatment_en": info["treatment_en"],
        "treatment_ta": info["treatment_ta"],
        "prevention_en": info["prevention_en"],
        "prevention_ta": info["prevention_ta"],
        "confidence": demo["confidence"],
        "source": "demo"
    })


# -------------------------------
# TEXT → AUDIO (TTS)
# -------------------------------
@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text given"}), 400

    try:
        tts = gTTS(text=text, lang="ta" if lang == "ta" else "en")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return send_file(fp, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": "TTS failed", "details": str(e)}), 500


# -------------------------------
# CHATBOT (FIXED VERSION)
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    lang = data.get("lang", "en")

    if not user_msg: # type: ignore
        return jsonify({"reply": "Empty message"}), 400

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    # ---- OPENAI ENABLED ----
    if OPENAI_KEY and openai is not None:
        try:
            # New SDK style
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI(api_key=OPENAI_KEY)
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful agriculture assistant."},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=300
                )
                reply = r.choices[0].message.content.strip() # type: ignore
                return jsonify({"reply": reply, "source": "openai"})

            # Old SDK (ChatCompletion)
            elif hasattr(openai, "ChatCompletion"):
                r = openai.ChatCompletion.create( # type: ignore
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful agriculture assistant."},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=300
                )
                reply = r.choices[0].message["content"].strip()
                return jsonify({"reply": reply, "source": "openai"})

        except Exception as e:
            print("OpenAI error:", e)

    # ---- FALLBACK CHATBOT ---- # type: ignore
    msg = user_msg.lower()

    # Yellow leaves
    if any(word in msg for word in ["yellow", "leaves yellow", "மஞ்சள்", "இலை மஞ்சள்"]):
        reply = "Leaves turning yellow may indicate nutrient deficiency (N, Mg) or overwatering."
        if lang == "ta":
            reply = "இலைகள் மஞ்சள் ஆகுவது பொதுவாக நைட்ரஜன் அல்லது மக்னீசியம் குறைபாடு, அல்லது அதிக நீர்ப்பாசனம் காரணமாக ஏற்படும்."

    # Dry leaves / burnt leaves
    elif any(word in msg for word in ["dry", "dried", "burnt", "brown tips", "உலர்", "கருகி"]):
        reply = "Dry or burnt leaves could be due to heat stress or low watering."
        if lang == "ta":
            reply = "இலைகள் உலர்வது அல்லது கருகுவது அதிக வெப்பம் அல்லது போதிய நீர் இல்லாமை காரணமாக இருக்கலாம்."

    # Fungus infection
    elif any(word in msg for word in ["fungus", "white powder", "mold", "பூஞ்சை", "வெள்ளை தூள்"]):
        reply = "White powdery substance indicates fungal infection. Use organic fungicide."
        if lang == "ta":
            reply = "வெள்ளை தூள் போன்றது பூஞ்சை நோய். கரிம பூஞ்சைக்கொல்லி பயன்படுத்தவும்."

    # Pest / insects
    elif any(word in msg for word in ["pest", "insect", "bugs", "worm", "பூச்சி", "கிருமி"]):
        reply = "Common pests can be controlled using neem oil spray every 3 days."
        if lang == "ta":
            reply = "பொதுவான பூச்சிகளை நீக்க 3 நாட்களுக்கு ஒரு முறை நீம் எண்ணெய் தெளிக்கவும்."

    # Black spots
    elif any(word in msg for word in ["black spot", "spots", "கருப்பு புள்ளி"]):
        reply = "Black spots may be caused by leaf spot disease. Remove infected leaves."
        if lang == "ta":
            reply = "இலையில் கருப்பு புள்ளிகள் 'லீஃப் ஸ்பாட்' நோயாக இருக்கலாம். பாதிக்கப்பட்ட இலைகளை அகற்றவும்."

    # Leaves falling
    elif any(word in msg for word in ["leaf fall", "leaves dropping", "இலை உதிர்வு"]):
        reply = "Leaf fall may happen due to stress, underwatering, or fungal disease."
        if lang == "ta":
            reply = "இலை உதிர்வு தாவரப் பாதிப்பு, குறைபட்ட நீர் அல்லது பூஞ்சை நோயால் ஏற்படும்."

    # Watering doubt
    elif any(word in msg for word in ["how to water", "watering", "நீர்", "நீர்ப்பாசனம்"]):
        reply = "Water only when top soil is dry. Avoid overwatering."
        if lang == "ta":
            reply = "மண் மேல் பகுதி உலர்ந்ததும் மட்டும் நீர் ஊற்றவும். அதிக நீர் தவிர்க்கவும்."

    # Fertilizer doubt
    elif any(word in msg for word in ["fertilizer", "feed", "compost", "உர", "கம்போஸ்ட்"]):
        reply = "Use organic compost every 15 days for healthy plant growth."
        if lang == "ta":
            reply = "ஒவ்வொரு 15 நாட்களிலும் கம்போஸ்ட் உரம் சேர்ப்பது தாவர வளர்ச்சியை மேம்படுத்தும்."

    # Growth problem
    elif any(word in msg for word in ["not growing", "slow growth", "வளரவில்லை", "மந்தமான வளர்ச்சி"]):
        reply = "Slow growth may indicate poor soil nutrients or low sunlight."
        if lang == "ta":
            reply = "வளர்ச்சி மந்தமாக இருப்பது உணவு குறைபாடு அல்லது குறைந்த சூரிய ஒளி காரணமாக இருக்கலாம்."

    # Water logging issue
    elif any(word in msg for word in ["water log", "too much water", "அதிக நீர்"]):
        reply = "Waterlogged soil can cause root rot. Improve drainage."
        if lang == "ta":
            reply = "அதிக நீர் வேர் சிதைவுக்கு காரணம். வடிகால் வசதி மேம்படுத்தவும்."

    # Sunlight requirement
    elif any(word in msg for word in ["sunlight", "light", "சூரிய ஒளி"]):
        reply = "Most plants need at least 4–6 hours of sunlight daily."
        if lang == "ta":
            reply = "அதிகாংশ தாவரங்களுக்கும் தினமும் 4-6 மணி நேர சூரிய ஒளி அவசியம்."

    # Soil doubt
    elif any(word in msg for word in ["soil", "sand", "pot mix", "மண்", "பாட்டிங் மிஷ்"]):
        reply = "Use well-drained loamy soil mixed with compost for best results."
        if lang == "ta":
            reply = "நல்ல வடிகால் கொண்ட களிமண் + கம்போஸ்ட் கலந்த மண் சிறந்தது."

    # Temperature concern
    elif any(word in msg for word in ["temperature", "heat", "cold", "வெப்பம்", "சூடு", "குளிர்"]):
        reply = "Extreme heat or cold can affect plant health. Provide shade or protection."
        if lang == "ta":
            reply = "அதிக சூடு அல்லது குளிர் தாவர ஆரோக்கியத்தை பாதிக்கும். நிழல் அல்லது பாதுகாப்பு அளிக்கவும்."

    # General fallback
    else:
        reply = "Ask about plant disease, symptoms, watering, soil, or pest control."
        if lang == "ta":
            reply = "தாவர நோய், அறிகுறி, நீர்ப்பாசனம், மண் அல்லது பூச்சி கட்டுப்பாடு பற்றி கேளுங்கள்."

    return jsonify({"reply": reply, "source": "fallback"})


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
