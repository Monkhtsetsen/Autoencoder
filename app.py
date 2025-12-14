import os
import io
import base64
import math

from flask import Flask, request, render_template_string
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

from model import UNetDenoiser  # UNet-DAE model

# ---- basic config ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 28          # MNIST size
IN_CHANNELS = 1
LATENT_DIM = 256       # bottleneck pooled to 256-D vector
MODEL_TYPE = "unet_dae"

WEIGHTS_PATH = "checkpoints_unet_dae/unet_dae_best.pth"
CURVE_PATH = "training_curve.png"   # байвал харагдана, байхгүй бол зүгээр алгасна
SAMPLES_PATH = "samples.png"        # демо samples зураг (хүсвэл)

app = Flask(__name__)

resize_to_model = T.Resize((IMG_SIZE, IMG_SIZE))
to_tensor = T.ToTensor()

# ---- load model ----
model = UNetDenoiser().to(device)
if os.path.exists(WEIGHTS_PATH):
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] Loaded weights from {WEIGHTS_PATH}")
else:
    print(f"[WARN] {WEIGHTS_PATH} not found. Run train.py first.")
model.eval()


# ------------- helpers -------------

def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def compute_metrics(x: torch.Tensor, recon: torch.Tensor) -> dict:
    mse = torch.mean((x - recon) ** 2).item()
    if mse <= 1e-12:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10(1.0 / mse)
    return {"mse": mse, "psnr": psnr}


def load_curve(path):
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    return pil_to_data_url(img)


def run_model_on_pil(pil_img: Image.Image):
    """
    Input: arbitrary PIL image
    Output: upscaled original/denoised/errormap PIL images + latent vector + metrics
    """
    # grayscale + resize
    gray = pil_img.convert("L")
    gray = resize_to_model(gray)

    x = to_tensor(gray)  # (1,H,W) in [0,1]

    # invert if white background (MNIST style)
    if x.mean().item() > 0.5:
        x = 1.0 - x

    x = x.unsqueeze(0).to(device)  # (1,1,H,W)

    with torch.no_grad():
        # UNetDenoiser forward(return_latent=True) -> (recon, latent)
        recon, z = model(x, return_latent=True)

    metrics = compute_metrics(x, recon)

    x_cpu = x[0].cpu().clamp(0.0, 1.0)
    recon_cpu = recon[0].cpu().clamp(0.0, 1.0)
    z_cpu = z[0].cpu()  # (LATENT_DIM,)

    # error map
    diff = (x_cpu - recon_cpu).abs()
    max_val = diff.max().item()
    if max_val > 1e-8:
        diff = diff / max_val

    # upscale for display
    upscale = T.Resize((256, 256))
    orig_pil = upscale(to_pil_image(x_cpu))
    recon_pil = upscale(to_pil_image(recon_cpu))
    diff_pil = upscale(to_pil_image(diff))

    return orig_pil, recon_pil, diff_pil, z_cpu, metrics


# ------------- HTML -------------


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ model_type }} Denoising ({{ img_size }}×{{ img_size }})</title>
  <style>
    :root {
      --pink-bg-light: #ffe6f2;
      --pink-bg-mid: #ffd6ea;
      --pink-bg-card: #fff0f6;
      --pink-border: #f9a8d4;
      --pink-soft: #ffe4f0;
      --pink-pill: #ffeaf3;
      --pink-text-main: #4b1f3f;
      --pink-text-soft: #9f1239;
      --pink-muted: #be185d;
      --pink-btn-from: #f9a8d4;
      --pink-btn-to: #fb7185;
      --pink-shadow: rgba(244, 114, 182, 0.45);
      --pink-error: #e11d48;
    }

    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, var(--pink-bg-light), var(--pink-bg-mid));
      color: var(--pink-text-main);
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      min-height: 100vh;
    }

    .container {
      max-width: 1100px;
      width: 100%;
      padding: 24px;
    }

    .card {
      background: var(--pink-bg-card);
      border-radius: 22px;
      padding: 22px 26px 26px;
      border: 1px solid var(--pink-border);
      box-shadow: 0 22px 60px var(--pink-shadow);
      backdrop-filter: blur(20px);
    }

    h1 {
      margin-top: 0;
      font-size: 24px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--pink-text-main);
    }

    p {
      color: var(--pink-text-soft);
      font-size: 14px;
      margin-top: 4px;
    }

    .pill-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }

    .pill {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      padding: 4px 10px;
      border-radius: 999px;
      background: var(--pink-pill);
      border: 1px solid var(--pink-border);
      color: var(--pink-muted);
    }

    .upload-box {
      margin-top: 18px;
      padding: 16px;
      border-radius: 16px;
      border: 1px dashed var(--pink-border);
      background: var(--pink-soft);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }

    input[type=file] {
      color: var(--pink-text-main);
      font-size: 14px;
    }

    .btn {
      padding: 9px 20px;
      border-radius: 999px;
      border: none;
      cursor: pointer;
      background: linear-gradient(135deg, var(--pink-btn-from), var(--pink-btn-to));
      color: #fff;
      font-weight: 600;
      font-size: 14px;
      box-shadow: 0 10px 25px var(--pink-shadow);
      white-space: nowrap;
      transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
    }

    .btn:hover {
      opacity: 0.96;
      transform: translateY(-1px);
      box-shadow: 0 14px 35px var(--pink-shadow);
    }

    .grid-main {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      margin-top: 22px;
    }

    .panel {
      background: #fff7fb;
      border-radius: 16px;
      border: 1px solid var(--pink-border);
      padding: 14px 14px 16px;
    }

    .panel h2 {
      font-size: 13px;
      font-weight: 600;
      margin: 0 0 8px 0;
      color: var(--pink-text-main);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .panel small {
      color: var(--pink-text-soft);
      font-size: 11px;
    }

    .img-wrapper {
      margin-top: 10px;
      display: flex;
      justify-content: center;
      align-items: center;
      background: radial-gradient(circle at top, #ffe4f0, #fed7e2);
      border-radius: 14px;
      overflow: hidden;
      min-height: 220px;
      border: 1px solid var(--pink-border);
    }

    .img-grid-3 {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      width: 100%;
    }

    .img-wrapper img {
      max-width: 100%;
      height: auto;
      display: block;
    }

    .metric-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }

    .metric-pill {
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #ffeaf3;
      border: 1px solid var(--pink-border);
      color: var(--pink-text-main);
    }

    .metric-pill span {
      color: var(--pink-muted);
      font-size: 11px;
      margin-right: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .latent-box {
      font-family: "JetBrains Mono", "Fira Code", monospace;
      font-size: 11px;
      background: #fff;
      border-radius: 12px;
      padding: 10px 12px;
      overflow-x: auto;
      border: 1px solid var(--pink-border);
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 260px;
      color: var(--pink-text-main);
    }

    details {
      margin-top: 8px;
    }

    summary {
      cursor: pointer;
      color: var(--pink-muted);
      font-size: 12px;
    }

    .status {
      margin-top: 10px;
      font-size: 12px;
      color: var(--pink-text-soft);
    }

    .error {
      margin-top: 10px;
      font-size: 12px;
      color: var(--pink-error);
      font-weight: 500;
    }

    .curve-panel {
      margin-top: 16px;
    }

    .download-row {
      margin-top: 12px;
      display: flex;
      justify-content: flex-end;
    }

    .btn-secondary {
      padding: 7px 16px;
      border-radius: 999px;
      border: 1px solid var(--pink-border);
      background: #ffeaf3;
      color: var(--pink-text-main);
      font-size: 13px;
      cursor: pointer;
      text-decoration: none;
      transition: background 0.12s ease, transform 0.12s ease;
    }

    .btn-secondary:hover {
      background: #ffd6ea;
      transform: translateY(-1px);
    }

    @media (max-width: 900px) {
      .grid-main {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>{{ model_type|upper }} Denoising – {{ img_size }}×{{ img_size }}</h1>
      <p>Blurry / noisy бичмэл цифр эсвэл handwriting-ийн жижиг patch ({{ img_size }}×{{ img_size }})
         upload хийгээд autoencoder хэрхэн цэвэрлэж байгааг харах demo.</p>

      <div class="pill-row">
        <div class="pill">Latent: {{ latent_dim }}D</div>
        <div class="pill">Compression: ~{{ compression_hint }}×</div>
        <div class="pill">Device: {{ device_name }}</div>
      </div>

      <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-box">
          <div>
            <input type="file" name="image" accept="image/*" required />
          </div>
          <button class="btn" type="submit">Denoise Image</button>
        </div>
      </form>

      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}

      {% if orig_img and recon_img %}
        <div class="grid-main">
          <div class="panel">
            <h2>ORIGINAL vs DENOISED vs ERROR MAP</h2>
            <small>Original (resize), model output, болон |x − x̂|-ийн normalized heatmap.</small>
            <div class="img-wrapper img-grid-3">
              <img src="{{ orig_img }}" alt="original" />
              <img src="{{ recon_img }}" alt="denoised" />
              <img src="{{ diff_img }}" alt="error map" />
            </div>
            <div class="metric-row">
              <div class="metric-pill"><span>MSE</span> {{ "%.6f"|format(mse) }}</div>
              <div class="metric-pill"><span>PSNR</span> {{ "%.2f"|format(psnr) }} dB</div>
            </div>
            <div class="status">
              Error map дээр цайвар хэсэг = алдаа ихтэй пиксел, бараан хэсэг = сайн reconstruction.
            </div>
            <div class="download-row">
              <a class="btn-secondary" href="{{ recon_img }}" download="denoised.png">
                Denoised зураг татаж авах
              </a>
            </div>
          </div>

          <div class="panel">
            <h2>Latent Encoding</h2>
            <small>{{ model_type|upper }} bottleneck ({{ latent_dim }} хэмжээтэй vector)</small>
            <details>
              <summary>Latent vector-ыг дэлгэж харах</summary>
              <div class="latent-box">
{{ latent_str }}
              </div>
            </details>
            <div class="status">
              UNet-DAE нь latent feature-ээрээ noisy ба clean дүрсний бүтцийг ялгаж сурсан байна.
            </div>
          </div>
        </div>
      {% else %}
        <div class="status">Зураг upload хийгээд доорх хэсэг автоматаар дүүрнэ.</div>
      {% endif %}

      {% if curve_img %}
        <div class="panel curve-panel">
          <h2>Training Curve (MNIST {{ img_size }}×{{ img_size }})</h2>
          <small>Train / Valid loss per epoch</small>
          <div class="img-wrapper" style="min-height:180px;">
            <img src="{{ curve_img }}" alt="training curve" />
          </div>
        </div>
      {% endif %}

      {% if samples_img %}
        <div class="panel curve-panel">
          <h2>Validation Samples</h2>
          <small>Noisy → Denoised → Clean (MNIST)</small>
          <div class="img-wrapper" style="min-height:180px;">
            <img src="{{ samples_img }}" alt="samples" />
          </div>
        </div>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""


# ------------- routes -------------

@app.route("/", methods=["GET"])
def index():
    compression_hint = round((IMG_SIZE * IMG_SIZE * IN_CHANNELS) / LATENT_DIM, 1)
    return render_template_string(
        HTML_PAGE,
        orig_img=None,
        recon_img=None,
        diff_img=None,
        mse=0.0,
        psnr=0.0,
        latent_str="",
        error=None,
        latent_dim=LATENT_DIM,
        img_size=IMG_SIZE,
        device_name=str(device),
        model_type=MODEL_TYPE,
        compression_hint=compression_hint,
        curve_img=load_curve(CURVE_PATH),
        samples_img=load_curve(SAMPLES_PATH),
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        error = "Файл олдоогүй. Зураг сонгоод дахин оролдоорой."
        return _render_with_error(error)

    file = request.files["image"]
    if file.filename == "":
        error = "Зураг сонгоогүй байна."
        return _render_with_error(error)

    try:
        pil_img = Image.open(file.stream)
    except Exception as e:
        return _render_with_error(f"Зургийг уншихад алдаа гарлаа: {e}")

    orig_pil, recon_pil, diff_pil, z_vec, metrics = run_model_on_pil(pil_img)

    orig_url = pil_to_data_url(orig_pil)
    recon_url = pil_to_data_url(recon_pil)
    diff_url = pil_to_data_url(diff_pil)

    latent_vals = [f"{v:.4f}" for v in z_vec.tolist()]
    chunks = [" ".join(latent_vals[i:i+8]) for i in range(0, len(latent_vals), 8)]
    latent_str = "\n".join(chunks)

    compression_ratio = round((IMG_SIZE * IMG_SIZE * IN_CHANNELS) / LATENT_DIM, 1)

    return render_template_string(
        HTML_PAGE,
        orig_img=orig_url,
        recon_img=recon_url,
        diff_img=diff_url,
        mse=metrics["mse"],
        psnr=metrics["psnr"],
        latent_str=latent_str,
        error=None,
        latent_dim=LATENT_DIM,
        img_size=IMG_SIZE,
        device_name=str(device),
        model_type=MODEL_TYPE,
        compression_hint=compression_ratio,
        curve_img=load_curve(CURVE_PATH),
        samples_img=load_curve(SAMPLES_PATH),
    )


def _render_with_error(error_msg: str):
    compression_hint = round((IMG_SIZE * IMG_SIZE * IN_CHANNELS) / LATENT_DIM, 1)
    return render_template_string(
        HTML_PAGE,
        orig_img=None,
        recon_img=None,
        diff_img=None,
        mse=0.0,
        psnr=0.0,
        latent_str="",
        error=error_msg,
        latent_dim=LATENT_DIM,
        img_size=IMG_SIZE,
        device_name=str(device),
        model_type=MODEL_TYPE,
        compression_hint=compression_hint,
        curve_img=load_curve(CURVE_PATH),
        samples_img=load_curve(SAMPLES_PATH),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
