from PIL import Image
import time
import torch
import logging
from datetime import datetime
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Configure processing logging
logging.basicConfig(
    filename='output/time/processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure output logging
output_logger = logging.getLogger('output_logger')
output_logger.setLevel(logging.INFO)
# Prevent output logger from using the basic config
output_logger.propagate = False
# Create file handler for output logging
output_handler = logging.FileHandler('output/output.log')
output_handler.setFormatter(logging.Formatter('%(asctime)s\n%(message)s\n%(separator)s\n', 
                                            datefmt='%Y-%m-%d %H:%M:%S'))
output_logger.addHandler(output_handler)




# Decorator to measure and log execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        
        # Get the image details from the global variables
        image_info = getattr(wrapper, 'image_info', {})
        
        # Create log message with image details
        log_message = (
            f"\nFunction: {func.__name__}\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Image Name: {image_info.get('image_name', 'N/A')}\n"
            f"Resized Image: {image_info.get('resized_image_name', 'N/A')}\n"
            f"Original Resolution: {image_info.get('original_resolution', 'N/A')}\n"
            f"Resized Resolution: {image_info.get('resized_resolution', 'N/A')}\n"
            f"Execution Time: {elapsed_time_seconds:.2f} seconds ({elapsed_time_minutes:.2f} minutes)\n"
            f"{'-'*50}"
        )
        
        # Log to file
        logging.info(log_message)
        
        # Print to console
        print(log_message)
        return result
    return wrapper





# Function to reduce image resolution by 20%
def resize_image(image_path, output_path):
    image = Image.open(image_path)
    original_resolution = f"{image.width}x{image.height}"
    new_width = int(image.width * 0.08)
    new_height = int(image.height * 0.08)
    resized_image = image.resize((new_width, new_height))
    resized_image.save(output_path)
    
    # Store image information
    image_info = {
        'image_name': image_path.split('/')[-1],
        'resized_image_name': output_path.split('/')[-1],
        'original_resolution': original_resolution,
        'resized_resolution': f"{new_width}x{new_height}"
    }
    
    # Attach image info to the wrapper function
    process_image.image_info = image_info
    
    print(f"Image resized to {new_width}x{new_height} and saved as {output_path}")
    return output_path





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
    
    # Resize image before processing
    input_image_path = "images/KaysMenu.jpg"
    resized_image_path = "images/resized/KaysMenu_Resized_to_8.jpg"
    resized_image_path = resize_image(input_image_path, resized_image_path)
    
    # Load and process the resized image
    image = Image.open(resized_image_path)
    
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
    
    # Log output with metadata to output.log
    image_info = getattr(process_image, 'image_info', {})
    output_log_message = (
        f"Image Name: {image_info.get('image_name', 'N/A')}\n"
        f"Resized Image: {image_info.get('resized_image_name', 'N/A')}\n"
        f"Original Resolution: {image_info.get('original_resolution', 'N/A')}\n"
        f"Resized Resolution: {image_info.get('resized_resolution', 'N/A')}\n"
        f"Model Output:\n{output_text}"
    )
    output_logger.info(output_log_message, extra={'separator': '-'*50})
    
    print("Output saved to output_text.txt and output.log")





# Entry point
if __name__ == "__main__":
    process_image()