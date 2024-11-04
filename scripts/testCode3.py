import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())

model_id = "C:\\codes\\llamaVision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}
)
processor = AutoProcessor.from_pretrained(model_id)

url = "images/KaysHours.jpg"
image = Image.open(url)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Given this image, output the information you see into a json format please"}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=2000)
print(processor.decode(output[0]))