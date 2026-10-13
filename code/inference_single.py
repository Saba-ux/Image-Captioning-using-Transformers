import json
import torch
from PIL import Image
from torchvision import transforms

from dataset.simple_vocab import SimpleVocab
from models.cnn_encoder import ImageEncoder
from models.IC_encoder_decoder.transformer import Transformer

DEVICE = torch.device("cpu")

IMAGE_PATH = "images/test.jpeg"
VOCAB_PATH = "data/mscoco_h5/vocab.json"
CHECKPOINT_PATH = "code/checkpoints/0310.1454/checkpoint.pth.tar"

vocab = SimpleVocab.load(VOCAB_PATH)

print("Vocabulary size:", len(vocab))

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

print("Checkpoint epoch:", checkpoint["epoch"])
print("Number of model states:", len(checkpoint["models"]))

image_encoder = ImageEncoder(
    encode_size=14,
    embed_dim=512
)

image_encoder = image_encoder.to(DEVICE)
image_encoder.eval()

print("Image encoder loaded")

transformer = Transformer(
    vocab_size=len(vocab),
    d_model=512,
    img_encode_size=196,
    enc_ff_dim=512,
    dec_ff_dim=2048,
    enc_n_layers=2,
    dec_n_layers=4,
    enc_n_heads=8,
    dec_n_heads=8,
    max_len=51,
    dropout=0.1,
    pad_id=vocab.stoi["<pad>"],  
)

transformer = transformer.to(DEVICE)
transformer.eval()

print("Transformer loaded")

image_encoder.load_state_dict(checkpoint["models"][0])
transformer.load_state_dict(checkpoint["models"][1])

print("Trained weights loaded successfully")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

print("Image loaded:", image.shape)

# ---------------------------------------------------------
# Generate caption using greedy decoding
# ---------------------------------------------------------
START_TOKEN = "<sos>"
END_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

start_id = vocab.stoi[START_TOKEN]
end_id = vocab.stoi[END_TOKEN]


with torch.no_grad():

    # Encode the image
    encoder_out = image_encoder(image)

    print("Encoder output shape:", encoder_out.shape)

    # Start caption with <start>
    caption = [start_id]

    max_length = 30

    for _ in range(max_length):

        caption_tensor = torch.tensor(
            caption,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        # Transformer expects:
        # images   -> [batch, 196, 512]
        # captions -> [batch, sequence_length]
        predictions, _ = transformer(
            encoder_out,
            caption_tensor
        )

        # Prediction for the newest word
        next_token = predictions[:, -1, :].argmax(dim=-1).item()

        # Stop at <end>
        if next_token == end_id:
            break

        caption.append(next_token)


# Convert token IDs back to words
words = []

for token_id in caption[1:]:
    word = vocab.itos[token_id]

    if word in [START_TOKEN, END_TOKEN, PAD_TOKEN]:
        continue

    words.append(word)


generated_caption = " ".join(words)

print()
print("=" * 60)
print("Generated caption:")
print(generated_caption)
print("=" * 60)