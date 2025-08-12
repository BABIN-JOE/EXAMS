!pip install transformers torch torchvision pillow


from google.colab import files
uploaded = files.upload()


image_file = list(uploaded.keys())[0]
print("✅ Uploaded file name:", image_file)


from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Load model and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Parameters (greedy decoding, no beam search)
gen_kwargs = {"max_length": 16, "num_return_sequences": 1}

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        images.append(image)

    # Feature extraction
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate captions (greedy decoding)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [caption.strip() for caption in captions]

# Upload and run prediction
from google.colab import files
uploaded = files.upload()

image_file = list(uploaded.keys())[0]
captions = predict_step([image_file])
print("🖼️ Caption:", captions[0])
img = Image.open(image_file)
plt.imshow(img)
plt.axis('off')
plt.title("Uploaded Image")
plt.show()
