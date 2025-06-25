import torch
from open_clip import tokenize
from model import clip_model,preprocess,device,base_model
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from eval import get_img

def get_img_features():
    cache_file= "cache_file(new).npy"
    if os.path.exists(cache_file):
        image_features=np.load(cache_file,allow_pickle=True).tolist()
        return image_features

def search(text_query, model, top_k=5):
    text_token = tokenize([text_query]).to(device)
    image_features = get_img_features()
    img=get_img(text_query)
    #dummy_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
    dummy_img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        #text_feat=base_model.encode_text(text_token)
        text_feat = model.encode_text(text_token, dummy_img_tensor)
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
        results=search(query,clip_model,top_k=10)
        num_results = len(results)
        cols = min(num_results, 10)
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