from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the auth token from environment variable
auth_token = os.getenv('USE_AUTH_TOKEN')

snapshot_download(
    repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    use_auth_token=auth_token,
    local_dir="\\wsl.localhost\\Ubuntu-20.04\\home\\anton\\projects\\MODELS\\llamaVision"
)