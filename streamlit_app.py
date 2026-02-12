import streamlit as st
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from ultralytics import YOLO
from PIL import Image

# =============================
# Klassen
# =============================
CLASS_NAMES = [
    "BACKGROUND",
    "BW4Cockpit (Stammdaten)",
    "DTP",
    "Data Flow Object",
    "Data Source",
    "Data Store Object",
    "Datenvorschau",
    "Excel",
    "Felder",
    "Formel",
    "Irrelevant",
    "Merkmal Eigenschaften",
    "Query",
    "Transformationen",
]

# =============================
# SSD laden
# =============================
@st.cache_resource
def load_ssd():
    model = ssd300_vgg16(weights=None)
    model.head.classification_head.num_classes = len(CLASS_NAMES)
    model.load_state_dict(
        torch.load("ssd_model_epochs10.pth", map_location="cpu")
    )
    model.eval()
    return model

# =============================
# YOLO laden
# =============================
@st.cache_resource
def load_yolo():
    return YOLO(
        "best.pt"
    )

# =============================
# UI
# =============================
st.title("Object Detection Demo – SSD vs YOLO")
st.write("Screenshot hochladen und Modell auswählen")

model_choice = st.selectbox(
    "Modell auswählen",
    [
        "SSD",
        "YOLOv8"
    ]
)

uploaded_file = st.file_uploader(
    "Bild auswählen", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Eingabebild", use_column_width=True)

    st.subheader("Ergebnis:")

    # =============================
    # SSD
    # =============================
    if model_choice == "SSD":
        model = load_ssd()

        img_tensor = torchvision.transforms.ToTensor()(image)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)[0]

        scores = output["scores"]
        labels = output["labels"]

        found = False
        for score, label in zip(scores, labels):
            if score.item() > 0.5:
                found = True
                st.write(
                    f"- **{CLASS_NAMES[label]}** "
                    f"(Confidence: {score.item():.2f})"
                )

        if not found:
            st.write("SSD hat keine Klasse mit ausreichender Sicherheit erkannt.")

    # =============================
    # YOLO
    # =============================
    else:
        yolo_model = load_yolo()

        # niedrigere Confidence, damit überhaupt Ergebnisse sichtbar werden
        results = yolo_model(image, conf=0.1)

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.write("YOLO hat keine Objekte über der Confidence-Schwelle erkannt.")
        else:
            st.write(f"{len(boxes)} Objekte erkannt:")

            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)

                # +1 wegen BACKGROUND bei SSD
                st.write(
                    f"- **{CLASS_NAMES[cls_id + 1]}** "
                    f"(Confidence: {conf:.2f})"
                )
