import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14'
                                                                  , pretrained='openai', device=device)

model_path = "./clip_model.pt"
checkpoints = torch.load(model_path, map_location=device)
base_model.load_state_dict(checkpoints['model_state_dict'])
for param in base_model.parameters():
    param.requires_grad = False


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v):
        attn_output, _ = self.multihead_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        q = q + self.dropout(attn_output)
        q = self.norm1(q)

        q2 = self.mlp(q)
        q = q + self.dropout(q2)
        q = self.norm2(q)
        return q


class CLIPCrossAttentionWrapper(nn.Module):
    def __init__(self, clip_model, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.clip_model = clip_model
        self.clip_embed_dim = embed_dim
        self.img_to_text_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.img_proj = nn.Linear(self.clip_model.visual.output_dim, embed_dim)
        self.text_proj = nn.Linear(self.clip_model.transformer.resblocks[-1].attn.out_proj.out_features, embed_dim)
        self.final_img_proj = nn.Linear(embed_dim, self.clip_model.visual.output_dim)
        self.logit_scale = self.clip_model.logit_scale

    def forward(self, images, texts):
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)

        img_features_proj = self.img_proj(image_features.unsqueeze(1))
        text_features_proj = self.text_proj(text_features.unsqueeze(1))

        img_attended = self.img_to_text_attn(img_features_proj, text_features_proj, text_features_proj)

        img_features_final = self.final_img_proj(img_attended.squeeze(1))

        image_features = F.normalize(img_features_final, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale

    def encode_image(self, images, texts):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(texts)
            img_features_proj = self.img_proj(image_features.unsqueeze(1))
            text_features_proj = self.text_proj(text_features.unsqueeze(1))
            img_attended = self.img_to_text_attn(img_features_proj, text_features_proj, text_features_proj)
            img_features_final = self.final_img_proj(img_attended.squeeze(1))
            return F.normalize(img_features_final, dim=-1)

    def encode_text(self, texts):
        with torch.no_grad():
            text_features = self.clip_model.encode_text(texts)
            text_features_proj = self.text_proj(text_features.unsqueeze(1))
            return F.normalize(text_features, dim=-1)


clip_model = CLIPCrossAttentionWrapper(
    clip_model=base_model,
).to(device)

model_path = "./one_cross_attention_model.pt"
checkpoints = torch.load(model_path, map_location=device)
clip_model.load_state_dict(checkpoints['model_state_dict'])
base_model.eval()
clip_model.eval()


from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json
with open("./output.json","r",encoding="utf-8") as f:
    data=json.load(f)
image_dir="./images/images/"
image_paths=[]
for item in data:
    image_paths.append(image_dir+item["image"])

from tqdm import tqdm
from PIL import Image
import numpy as np
from open_clip import tokenize


def get_img_features(model):
    cache_file = "./cache_file(one_cross).npy"
    if os.path.exists(cache_file):
        image_features = np.load(cache_file, allow_pickle=True).tolist()
    else:
        image_features = []
        for idx, path in enumerate(tqdm(image_paths)):
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            text_tensor = tokenize([data[idx]["caption"]]).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_tensor, text_tensor)
                image_features.append(("./images/images" + path[27:], feat.cpu().numpy()))
        np.save(cache_file, np.array(image_features, dtype=object))
    return image_features


def search(text_query, model, top_k=5):
    text_token = tokenize([text_query]).to(device)
    image_features = get_img_features(model)

    with torch.no_grad():
        text_feat = model.encode_text(text_token)
        text_feat = text_feat.cpu().numpy()
    similarities = []
    for img_path, img_feat in image_features:
        sim = np.dot(text_feat, img_feat.T) / (np.linalg.norm(text_feat) * np.linalg.norm(img_feat))
        similarities.append((img_path, sim[0][0]))
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


if __name__ =="__main__":
    while True:
        query=input("请输入文本")
        results=search(query,clip_model)
        num_results = len(results)
        cols = min(num_results, 5)
        rows = (num_results + cols - 1) // cols
        plt.figure(figsize=(4 * cols, 4 * rows))
        for i, (path, sim) in enumerate(results, 1):
            img = Image.open(path).convert("RGB")
            plt.subplot(rows, cols, i)
            plt.imshow(img)
            plt.title(f"Similarity: {sim:.4f}", fontsize=10)
            plt.axis("off")
        plt.suptitle(f"resluts: \"{query}\"", fontsize=16)
        plt.tight_layout()
        plt.show()