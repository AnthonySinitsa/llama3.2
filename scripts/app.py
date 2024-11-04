import time
import torch
import json
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Decorator to measure and log execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        with open("output/time/time_log.txt", "a") as log_file:
            log_file.write(f"{func.__name__} elapsed time: {elapsed_time_seconds:.2f} seconds "
                           f"({elapsed_time_minutes:.2f} minutes)\n")
        print(f"{func.__name__} elapsed time: {elapsed_time_seconds:.2f} seconds "
              f"({elapsed_time_minutes:.2f} minutes)")
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
    url = "images/KaysHours3.jpg"
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
    output_text = processor.decode(output[0])

    # Save output to a .txt file
    with open("output/output_text.txt", "w") as text_file:
        text_file.write(output_text)
    
    # Save output to a .json file
    try:
        # Attempt to parse the output as JSON
        output_json = json.loads(output_text)
    except json.JSONDecodeError:
        # If output is not valid JSON, save as plain text in JSON format
        output_json = {"response": output_text}
    
    with open("output/json/output_data.json", "w") as json_file:
        json.dump(output_json, json_file, indent=4)
    
    print("Output saved to output_text.txt and output_data.json")

# Entry point
if __name__ == "__main__":
    process_image()
