from segment_anything import sam_model_registry
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
from einops import rearrange


class FTCLIP(nn.Module):
    def __init__(self, sam_path, clip_path):
        super(FTCLIP, self).__init__()
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_path)
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=clip_path
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=clip_path
        )
        self.SAM_to_CLIP = nn.Linear(256, 3)

    def encode_image(self, image):
        with torch.no_grad():
            image_embedding = self.sam.image_encoder(image.squeeze(1))

        upscaled_embedding = torch.nn.functional.interpolate(
            image_embedding, size=(224, 224), mode="bilinear", align_corners=False
        )
        return upscaled_embedding

    def forward(self, pre_image, post_image, description):
        # encode image with SAM
        sam_emb_pre = self.encode_image(pre_image.cuda())
        sam_emb_post = self.encode_image(post_image.cuda())
        sam_emb_diff = sam_emb_pre - sam_emb_post
        sam_emb_diff = rearrange(sam_emb_diff, "B D H W -> B H W D")
        # change SAM dim to CLIP dim
        sam_emb_diff = self.SAM_to_CLIP(sam_emb_diff)
        sam_emb_diff = rearrange(sam_emb_diff, "B H W D -> B D H W")
        #
        inputs = self.clip_processor(
            text=description,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        pixel_values = sam_emb_diff.clone()
        input_ids = inputs["input_ids"].squeeze().cuda()
        attention_mask = inputs["attention_mask"].squeeze().cuda()
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits_per_image

        return logits
