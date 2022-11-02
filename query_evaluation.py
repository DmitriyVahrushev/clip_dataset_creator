from PIL import Image
import numpy as np
import torch
import clip


class QueryEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def evaluate(self, image, text_query, bboxes):
        images = []
        for i in range(bboxes.xyxy[0].shape[0]):
            bb_coords = bboxes.xyxy[0][i].cpu().numpy().astype('int')
            bb_img = image.copy()
            bb_img[:,:,:] = 0
            bb_img[bb_coords[1]:bb_coords[3],bb_coords[0]:bb_coords[2]] = image[bb_coords[1]:bb_coords[3],bb_coords[0]:bb_coords[2]]
            im = Image.fromarray(bb_img)
            image1 = self.preprocess(im)
            images.append(image1)
        imgs_tensor = torch.stack(images).to(self.device)
        text = clip.tokenize([text_query]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(imgs_tensor)
            text_features = self.model.encode_text(text)
            logits_per_image, logits_per_text = self.model(imgs_tensor, text)
            probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        return probs