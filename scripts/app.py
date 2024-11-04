import time
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Decorator to measure and log execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open("time_log.txt", "a") as log_file:
            log_file.write(f"{func.__name__} elapsed time: {elapsed_time:.2f} seconds\n")
        print(f"{func.__name__} elapsed time: {elapsed_time:.2f} seconds")
        return result
    return wrapper

@measure_time
def process_image():
    print("CUDA available: ", torch.cuda.is_available())
    print("CUDA device count: ", torch.cuda.device_count())
    
    model_id = "C:\\codes\\llamaVision"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load and process the image
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
    
    # Generate output and decode
    output = model.generate(**inputs, max_new_tokens=2000)
    print(processor.decode(output[0]))

# Entry point
if __name__ == "__main__":
    process_image()
