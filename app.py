import sys
import subprocess

try:
    import gradio as gr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from glob import glob

MODEL_PATH = None
for candidate in ['best.pt', 'runs/segment/train/weights/best.pt']:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    found = glob('**/best.pt', recursive=True)
    if found:
        MODEL_PATH = found[-1]

if MODEL_PATH is None:
    print('ERROR: best.pt not found. Place it in the same folder as this script.')
    model = None
else:
    print(f'Loaded model from: {MODEL_PATH}')
    model = YOLO(MODEL_PATH)


def predict_land_cover(input_image, confidence_threshold):
    if input_image is None or model is None:
        return None
    results = model(input_image, conf=confidence_threshold)
    result = results[0]
    if result.masks is None:
        return input_image
    plotted_bgr = result.plot(boxes=False, labels=False, conf=False)
    plotted_rgb = np.asarray(plotted_bgr)[..., ::-1].copy()
    return Image.fromarray(plotted_rgb)


custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    background_fill_primary="#f0fdf4",
    background_fill_primary_dark="#06150e"
)

custom_css = """
#gradient-header {
    background: linear-gradient(135deg, #059669, #064e3b);
    color: white !important;
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}
#gradient-header h1, #gradient-header p, #gradient-header strong { color: white !important; }
.section-title h3 {
    margin-top: 0px !important;
    padding-bottom: 8px;
    border-bottom: 2px solid #10b981;
    display: inline-block;
}
"""

with gr.Blocks(theme=custom_theme, css=custom_css, title="DeepGlobe YOLOv8-seg") as demo:

    gr.Markdown(
        """
        # DeepGlobe Land Cover Classification
        **Supervised Satellite Segmentation Platform**

        Interactive interface running custom-trained YOLOv8-seg weights on DeepGlobe and LandCover.ai satellite imagery.
        """,
        elem_id="gradient-header"
    )

    with gr.Row():
        with gr.Column(scale=11):
            gr.Markdown("### Upload Image", elem_classes=["section-title"])
            input_img = gr.Image(type="pil", label="Upload Satellite Image Tile")
            gr.Markdown("### Inference Settings", elem_classes=["section-title"])
            conf_slider = gr.Slider(
                minimum=0.01, maximum=1.0, value=0.25, step=0.05,
                label="Confidence Threshold",
                info="Lower = more detections, Higher = only confident predictions"
            )
            submit_btn = gr.Button("Analyze Land Cover", variant="primary", size="lg")

        with gr.Column(scale=1):
            pass

        with gr.Column(scale=11):
            gr.Markdown("### Segmentation Results", elem_classes=["section-title"])
            output_img = gr.Image(type="pil", label="YOLOv8-seg Output", interactive=False)

            with gr.Accordion("Land Cover Classes", open=True):
                gr.HTML("""
                <div style="padding: 10px;">
                    <div style="display:flex; align-items:center; margin-bottom:8px;">
                        <div style="width:20px;height:20px;background:#00FFFF;border:1px solid #ccc;margin-right:10px;border-radius:3px;"></div>
                        <span><strong>urban_land</strong> — Built-up areas, housing, infrastructure</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:8px;">
                        <div style="width:20px;height:20px;background:#FFFF00;border:1px solid #ccc;margin-right:10px;border-radius:3px;"></div>
                        <span><strong>agriculture</strong> — Crop fields and cultivated lands</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:8px;">
                        <div style="width:20px;height:20px;background:#FF00FF;border:1px solid #ccc;margin-right:10px;border-radius:3px;"></div>
                        <span><strong>rangeland</strong> — Meadows, shrublands, open fields</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:8px;">
                        <div style="width:20px;height:20px;background:#00FF00;border:1px solid #ccc;margin-right:10px;border-radius:3px;"></div>
                        <span><strong>forest</strong> — Dense tree canopies and woodlands</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:8px;">
                        <div style="width:20px;height:20px;background:#0000FF;border:1px solid #ccc;margin-right:10px;border-radius:3px;"></div>
                        <span><strong>water</strong> — Rivers, lakes, water channels</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:8px;">
                        <div style="width:20px;height:20px;background:#FFFFFF;border:1px solid #ccc;margin-right:10px;border-radius:3px;"></div>
                        <span><strong>barren_land</strong> — Exposed soil, sand, rocky terrains</span>
                    </div>
                </div>
                """)

    submit_btn.click(fn=predict_land_cover, inputs=[input_img, conf_slider], outputs=output_img)

demo.launch(share=True)