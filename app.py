import sys
import subprocess
import streamlit as st

try:
    import torch
except Exception as e:
    st.error(f"Python: {sys.version}")
    st.error(f"Error asli: {e!r}")
    out = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout
    st.code(out)
    st.stop()
    
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
    device = torch.device("cpu")  # server deploy tidak punya GPU
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_NAMES))

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}  # jaga-jaga kalau disimpan dari DataParallel

    model.load_state_dict(state)
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
    if st.button("Prediksi", type="primary", width="stretch"):
        try:
            model, device = load_model()
            with torch.no_grad():
                pred = CLASS_NAMES[model(transform(img).unsqueeze(0).to(device)).argmax(1).item()]
            (st.success if pred == "normal" else st.warning)(f"**Hasil Prediksi: {pred.upper()}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")