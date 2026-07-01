import os
import zipfile
import tempfile
import gdown
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gtts import gTTS

st.set_page_config(page_title="Image Captioning for visually imparied", page_icon="🖼️", layout="centered")

MODEL_DIR = "saved_ViT-SLSTM_models"
ZIP_FILE = "saved_ViT-SLSTM_models.zip"
FILE_ID = "1rEXAKAZiHtvYYPRChiVmdDV4wDYFbLc3"

REQUIRED_FILES = [
    "config.pth",
    "caption_model.pth",
    "tokenizer.pth",
    "vit_feature_extractor.pth",
]


def models_exist():
    """Check whether all required model files exist."""
    return all(
        os.path.isfile(os.path.join(MODEL_DIR, file))
        for file in REQUIRED_FILES
    )


def download_models():
    """Download and extract pretrained models if not already available."""

    if models_exist():
        st.success("✅ Pretrained models already exist.")
        return

    st.info("📥 Downloading pretrained models...")

    # Direct Google Drive download URL
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    try:
        # Download ZIP file
        gdown.download(
            url=url,
            output=ZIP_FILE,
            quiet=False
        )

        # Verify download
        if not os.path.exists(ZIP_FILE):
            st.error("❌ Download failed. ZIP file not found.")
            st.stop()

        st.info("📦 Extracting model files...")

        # Extract ZIP
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")

        # Remove ZIP after extraction
        os.remove(ZIP_FILE)

        # Verify required files
        if models_exist():
            st.success("✅ Models downloaded and extracted successfully.")
        else:
            st.error("❌ Some required model files are missing after extraction.")
            st.stop()

    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        st.stop()


# Download models when app starts
download_models()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class ViTFeatureExtractor(nn.Module):
    def __init__(self,vit):
        super().__init__(); self.vit=vit
    def forward(self,x):
        n=x.shape[0]
        x=self.vit._process_input(x)
        cls=self.vit.class_token.expand(n,-1,-1)
        x=torch.cat([cls,x],1)
        x=self.vit.encoder(x)
        return x[:,0]

class SeparableCNNLSTM(nn.Module):

    def __init__(self, embed_dim=512, hidden_dim=512):
        super().__init__()

        self.depthwise = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim
        )

        self.pointwise = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=1
        )

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, x):

        x = x.permute(0,2,1)

        x = self.depthwise(x)

        x = self.pointwise(x)

        x = self.relu(x)

        x = x.permute(0,2,1)

        _, (hidden, _) = self.lstm(x)

        return hidden[-1]


class CaptionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.image_fc = nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.embedding = nn.Embedding(
            vocab_size,
            512
        )

        self.sep_cnn_lstm = SeparableCNNLSTM()

        self.fc1 = nn.Linear(512,512)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.4)

        self.fc2 = nn.Linear(
            512,
            vocab_size
        )

    def forward(self, image, seq):

        img = self.image_fc(image)

        emb = self.embedding(seq)

        text = self.sep_cnn_lstm(emb)

        x = img + text

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

config=torch.load(f"{MODEL_DIR}/config.pth",weights_only=False)
max_len=config["max_len"]
vocab_size=config["vocab_size"]
tokenizer=torch.load(f"{MODEL_DIR}/tokenizer.pth",weights_only=False)
index_word=tokenizer.index_word


@st.cache_resource
def load_models():
    vit=vit_b_16(weights="DEFAULT").to(device)
    fe=ViTFeatureExtractor(vit).to(device)
    fe.load_state_dict(torch.load(f"{MODEL_DIR}/vit_feature_extractor.pth",map_location=device,weights_only=False))
    fe.eval()
    model=CaptionModel().to(device)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/caption_model.pth",map_location=device,weights_only=False))
    model.eval()
    return fe,model

feature_extractor,model=load_models()

@torch.no_grad()
def predict(image):
    t=transform(image).unsqueeze(0).to(device)
    feat=feature_extractor(t)
    text="startseq"
    words=[]
    for _ in range(max_len):
        seq=tokenizer.texts_to_sequences([text])[0]
        seq=pad_sequences([seq],maxlen=max_len,padding="post")[0]
        seq=torch.tensor(seq,dtype=torch.long).unsqueeze(0).to(device)
        out=model(feat,seq)
        idx=torch.argmax(out,1).item()
        w=index_word.get(idx)
        if w is None or w=="endseq":
            break
        words.append(w)
        text+=" "+w
    return " ".join(words)

st.title("🖼️ Image Caption Generator")
st.write("Upload an image to generate a caption and speech.")

uploaded=st.file_uploader("Choose an image",type=["jpg","jpeg","png"])

if uploaded:
    img=Image.open(uploaded).convert("RGB")
    st.image(img,use_container_width=True)
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            cap=predict(img)
        st.success("Done!")
        st.subheader("Caption")
        st.write(cap)
        tts=gTTS(text=cap or "No caption generated",lang="en")
        tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3")
        tts.save(tmp.name)
        st.audio(tmp.name)
        with open(tmp.name,"rb") as f:
            st.download_button("Download Audio",f,"caption.mp3","audio/mpeg")
