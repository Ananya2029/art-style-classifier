import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ArtStyle Predictor",
    page_icon="🎨",
    layout="centered"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device("cpu")


# =========================================================
# CLASS NAMES
# IMPORTANT:
# This order must match the order used during training.
# ImageFolder normally sorts class folders alphabetically.
# =========================================================

class_names = [
    "Baroque",
    "Cubism",
    "Expressionism",
    "Impressionism",
    "Pop Art",
    "Post-Impressionism",
    "Primitivism",
    "Renaissance",
    "Romanticism",
    "Suprematism",
    "Surrealism",
    "Symbolism"
]


# =========================================================
# MODEL
# =========================================================

class SimpleArtClassifier(nn.Module):

    def __init__(self, num_classes=12):

        super(SimpleArtClassifier, self).__init__()

        self.base_model = timm.create_model(
            "efficientnet_b0",
            pretrained=True
        )

        self.features = nn.Sequential(
            *list(self.base_model.children())[:-1]
        )

        enet_out_size = 1280

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                enet_out_size,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        output = self.classifier(x)

        return output


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = SimpleArtClassifier(
        num_classes=len(class_names)
    )

    checkpoint = torch.load(
        "artifacts/model.pth",
        map_location=DEVICE
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    # Remove "module." prefix if model was trained with DataParallel
    new_state_dict = {}

    for key, value in checkpoint.items():

        new_key = key.replace("module.", "")

        new_state_dict[new_key] = value

    model.load_state_dict(
        new_state_dict,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# =========================================================
# LOAD STYLE DATA
# =========================================================

@st.cache_data
def load_style_data():

    df = pd.read_csv(
        "data/art_style.csv"
    )

    return df


style_data = load_style_data()


# =========================================================
# IMAGE TRANSFORMATION
# IMPORTANT:
# Must match the preprocessing used during training.
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(image_file):

    image = Image.open(
        image_file
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

    predicted_idx = predicted_idx.item()

    confidence = confidence.item()

    return predicted_idx, confidence, probabilities


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎨 ArtStyle Predictor")

uploaded_file = st.sidebar.file_uploader(
    "Choose an image...",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # DISPLAY IMAGE
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Artwork",
        width="stretch"
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    predicted_idx, confidence, probabilities = predict(
        uploaded_file
    )


    predicted_class = class_names[
        predicted_idx
    ]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.write("## 🎨 Predicted Art Style")

    st.title(
        predicted_class
    )


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    st.write(
        f"### Confidence: {confidence * 100:.2f}%"
    )

    st.progress(
        confidence
    )


    # -----------------------------------------------------
    # STYLE INFORMATION
    # -----------------------------------------------------

    matching_style = style_data[
        style_data["style"].str.strip().str.lower()
        ==
        predicted_class.strip().lower()
    ]


    if not matching_style.empty:

        style_info = matching_style.iloc[0]

        st.subheader(
            "About this Art Style"
        )

        st.write(
            style_info["description"]
        )


        # -------------------------------------------------
        # WIKIPEDIA
        # -------------------------------------------------

        if (
            "wiki_url" in style_info
            and
            pd.notna(style_info["wiki_url"])
        ):

            st.markdown(
                f"[📖 More about {predicted_class} on Wikipedia]"
                f"({style_info['wiki_url']})"
            )

    else:

        st.warning(
            f"No information found in art_style.csv "
            f"for '{predicted_class}'."
        )


    # -----------------------------------------------------
    # TOP 5 PREDICTIONS
    # -----------------------------------------------------

    st.subheader(
        "🔍 Top 5 Predictions"
    )

    probabilities = probabilities.squeeze()

    top5_probabilities, top5_indices = torch.topk(
        probabilities,
        min(5, len(class_names))
    )

    for probability, index in zip(
        top5_probabilities,
        top5_indices
    ):

        class_name = class_names[
            index.item()
        ]

        probability_value = (
            probability.item() * 100
        )

        st.write(
            f"**{class_name}** — "
            f"{probability_value:.2f}%"
        )

        st.progress(
            probability.item()
        )


else:

    # =====================================================
    # LANDING PAGE
    # =====================================================

    st.title(
        "🎨 ArtStyle Predictor"
    )

    st.subheader(
        "Upload an image of a painting "
        "to discover its artistic style!"
    )

    st.write(
        "ArtStyle Predictor is your personal "
        "art historian. Upload a painting and "
        "the model will predict its artistic "
        "style and provide information about "
        "that style."
    )

    import os
    if os.path.exists("assets/landing.png"):
        st.image(
            "assets/landing.png",
            width="stretch"
        )