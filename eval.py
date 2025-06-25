import torch
from open_clip import tokenize
from cv1.model import clip_model,base_model,preprocess,device
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path= "images/images"
image_paths=[os.path.join(image_path,f) for f in os.listdir(image_path)]

clip_model=clip_model
clip_model.eval()
base_model=base_model
base_model.eval()

def get_img_features(model):
    cache_file= "cache_file.npy"
    if os.path.exists(cache_file):
        image_features=np.load(cache_file,allow_pickle=True).tolist()
    else:
        image_features=[]
        for path in tqdm(image_paths):
            img=Image.open(path).convert("RGB")
            img_tensor=preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat=model.encode_image(img_tensor)
                image_features.append((path,feat.cpu().numpy()))
        np.save(cache_file,np.array(image_features,dtype=object))
    return image_features

def search(text_query,model,top_k=5):
    text_token=tokenize([text_query]).to(device)
    image_features=get_img_features(model)
    with torch.no_grad():
        text_feat = model.encode_text(text_token)
        text_feat = text_feat.cpu().numpy()
    similarities = []
    for img_path, img_feat in image_features:
        sim = np.dot(text_feat, img_feat.T) / (np.linalg.norm(text_feat) * np.linalg.norm(img_feat))
        similarities.append((img_path, sim[0][0]))
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]

def get_img(query=""):
    results = search(query, base_model)
    img=Image.open(results[0][0]).convert("RGB")
    return img

if __name__ =="__main__":
    while True:
        query=input("请输入文本")
        results=search(query,base_model)
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