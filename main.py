import base64
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model="openrouter:openrouter/free")

with open("./examples/cat.jpg", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

messages = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": image_data,
            "mime_type": "image/jpeg",
        },
    ],
}

response = model.invoke([messages])

print(response.content)