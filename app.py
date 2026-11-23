import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm

CLASS_NAMES = ['blast', 'blight', 'normal', 'tungro']
MODEL_PATH = "Model_V2.pth"

st.set_page_config(page_title="Klasifikasi Penyakit Daun Padi", page_icon="🌾", layout="centered")
st.markdown("""
<style>
header, footer, #MainMenu {display: none !important;}
.block-container {padding-top: 1.5rem; padding-bottom: 0; max-width: 600px;}
div[data-testid="stImage"] img {max-height: 220px; width: auto !important; margin: auto; display: block;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model '{MODEL_PATH}' tidak ditemukan di direktori aplikasi.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_NAMES))
    
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model.to(device).eval(), device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.markdown("<h2 style='text-align:center'>🌾 Klasifikasi Penyakit Daun Padi</h2>", unsafe_allow_html=True)

file = st.file_uploader("Pilih gambar daun padi", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)
    if st.button("Prediksi", type="primary", use_container_width=True):
        with st.spinner("Menganalisis gambar..."):
            try:
                model, device = load_model()
                with torch.no_grad():
                    tensor_img = transform(img).unsqueeze(0).to(device)
                    pred_idx = model(tensor_img).argmax(1).item()
                    pred = CLASS_NAMES[pred_idx]
                
                if pred == "normal":
                    st.success(f"**Hasil Prediksi: {pred.upper()}**")
                else:
                    st.warning(f"**Hasil Prediksi: {pred.upper()}**")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")