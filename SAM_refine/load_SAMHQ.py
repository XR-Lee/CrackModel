from segment_anything_hq import sam_model_registry,SamPredictor

def load_SAM():
    model_type = "vit_h" #"vit_l/vit_b/vit_h/vit_tiny"
    sam_checkpoint = "/home/jc/xinrun/SAM/segment-anything/weights/sam_hq_vit_h.pth"
    # sam_checkpoint = "<path/to/checkpoint>"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return predictor