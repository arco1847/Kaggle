#!pip install -U transformers
#!pip install gradio easyocr transformers accelerate bitsandbytes sentencepiece --quiet
#!pip install gradio easyocr transformers accelerate sentencepiece bitsandbytes --quiet
#!pip install --upgrade bitsandbytes transformers accelerate sentencepiece gradio easyocr --quiet
#!pip install -U bitsandbytes
#!pip install --upgrade bitsandbytes transformers accelerate gradio easyocr --quiet
#!pip install --upgrade transformers accelerate gradio easyocr --quiet
#!pip install --upgrade gradio

# ==============================
# 1. Import Libraries
# ==============================
import gradio as gr
import easyocr
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==============================
# 2. Force Device Separation
# ==============================

# OCR → CPU
reader = easyocr.Reader(['en', 'bn'], gpu=False)

# LLM → GPU if available
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# ==============================
# 3. Load Qwen LLM
# ==============================
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct"
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    #torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    torch_dtype= torch.float32,
    device_map="cpu",
    low_cpu_mem_usage = True
)

print("OCR running on CPU")
print("LLM running on:", model.device)


# ==============================
# 4. Image Preprocessing
# ==============================
def preprocess_image(image):

    img = np.array(image)

    # Convert RGB → Gray (FIXED)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Noise removal
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (better for NID cards)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        15
    )

    return gray, thresh, img


# ==============================
# 5. Connected Component Boxing
# ==============================
def connected_components_boxes(thresh_img, original_img):

    # Morphology → group characters into words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated,
        connectivity=8
    )

    boxed_img = original_img.copy()

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter noise
        if area > 500:

            cv2.rectangle(
                boxed_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    return boxed_img


# ==============================
# 6. OCR Function
# ==============================
def extract_text_easyocr(processed_img):

    results = reader.readtext(processed_img)

    extracted_lines = []

    for bbox, text, confidence in results:
        extracted_lines.append(text)

    return "\n".join(extracted_lines)


# ==============================
# 7. LLM Extraction Function
# ==============================
def extract_nid_info_with_llm(ocr_text):

    messages = [
        {
            "role": "system",
            "content":
"""
You extract structured data from Bangladesh NID OCR text.

Return JSON only.

Do NOT guess missing values.
Keep Bangla unchanged.
"""
        },
        {
            "role": "user",
            "content": f"""
Extract the following fields if present:

Name_Bangla
Name_English
Father_Name
Mother_Name
Date_of_Birth
NID_Number
Address
Blood_Group
Place_of_Birth
Issue_Date

OCR TEXT:
{ocr_text}
"""
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return response


# ==============================
# 8. Full Pipeline
# ==============================
def process_nid(image):

    gray, thresh, original = preprocess_image(image)

    # Connected component visualization
    boxed_img = connected_components_boxes(thresh, original)

    # OCR on gray (better than thresh for EasyOCR)
    ocr_text = extract_text_easyocr(gray)

    structured_info = extract_nid_info_with_llm(ocr_text)

    return boxed_img, ocr_text, structured_info


# ==============================
# 9. Gradio UI
# ==============================
interface = gr.Interface(

    fn=process_nid,

    inputs=gr.Image(type="pil", label="Upload NID Card"),

    outputs=[
        gr.Image(label="Connected Component Boxes"),
        gr.Textbox(label="OCR Extracted Text"),
        gr.Textbox(label="Structured Information (LLM JSON)")
    ],

    title="Bangladesh NID Card Information Extractor",

    description=(
        "Connected Components group text into blocks. "
        "OCR runs on CPU. LLM runs on GPU."
    )
)


# ==============================
# 10. Launch App
# ==============================
if __name__ == "__main__":
    interface.launch()