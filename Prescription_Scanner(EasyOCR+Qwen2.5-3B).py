!pip install -U transformers

!pip install gradio easyocr transformers accelerate bitsandbytes sentencepiece --quiet
!pip install gradio easyocr transformers accelerate sentencepiece bitsandbytes --quiet
!pip install --upgrade bitsandbytes transformers accelerate sentencepiece gradio easyocr --quiet
!pip install -U bitsandbytes
!pip install --upgrade bitsandbytes transformers accelerate gradio easyocr --quiet
!pip install --upgrade transformers accelerate gradio easyocr --quiet
!pip install --upgrade gradio


import gradio as gr
import easyocr
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#1️........OCR on CPU (Bangla + English)
reader = easyocr.Reader(['en', 'bn'], gpu=False)  # CPU OCR

#2️........Qwen2.5-3B on GPU safely
model_name = "Qwen/Qwen2.5-3B-Instruct"

#useing float16
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=0, 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

#3️.........Gradio function
def extract_prescription(image):
    img_array = np.array(image)

    try:
        extracted_text = " ".join(reader.readtext(img_array, detail=0))
    except Exception as e:
        return f"OCR failed: {e}"

    if len(extracted_text.strip()) == 0:
        return "No text detected by OCR."

    #LLM data extraction
    prompt = f"Extract structured medical prescription info from this text:\n{extracted_text}"

    try:
        llm_output = llm_pipeline(prompt)[0]['generated_text']
    except Exception as e:
        llm_output = f"LLM processing failed: {e}"

    return llm_output

#4️..........Gradio interface
iface = gr.Interface(
    fn=extract_prescription,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(lines=15),
    title="Prescription Extractor (OCR + Qwen2.5-3B)",
    description="Handwritten prescriptions (Bangla + English). OCR on CPU, Qwen2.5-3B on GPU with FP16 (no 8-bit)."
)


iface.launch(debug=True)
