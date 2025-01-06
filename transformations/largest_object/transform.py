import argparse
import glob
import numpy as np
import os
import random
import sys
import torch

from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

from PIL import Image, ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# ------------------------------------------------------------------------
# Adjust this import path if needed, depending on your folder structure
# ------------------------------------------------------------------------
current_file_path = os.path.abspath(__file__)
sys.path.append(
    os.path.join(
        os.sep,
        *current_file_path.split(os.sep)[: current_file_path.split(os.sep).index("understand_bias") + 1]
    )
)

from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils


class Image_Dataset(Dataset):
    """
    Loads images from IMAGE_ROOTS[args.dataset]/args.split,
    resizes them if min dimension > 500, then returns them as NumPy arrays.
    """
    def __init__(self, args):
        # We'll save to "depth" here, but you can change to another key if desired
        self.output_dir = os.path.join(SAVE_ROOTS['largest_object'], args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)

        self.root = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
        self.paths = (
            glob.glob(os.path.join(self.root, "*.jpg")) +
            glob.glob(os.path.join(self.root, "*.png")) +
            glob.glob(os.path.join(self.root, "*.JPEG"))
        )
        if args.num is not None:
            self.paths = self.paths[: args.num]
        assert len(self.paths) == args.num, (
            f"Not enough images in {args.dataset}/{args.split} split"
        )

        # Simple resize if min dimension > 500
        self.preprocess = transforms.Resize(
            500, interpolation=transforms.InterpolationMode.BICUBIC
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        save_path = os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(path))[0] + ".png"
        )

        try:
            # Load PIL image
            with open(path, 'rb') as f:
                image = Image.open(f).convert("RGB")

            if min(image.size) > 500:
                image = self.preprocess(image)

            # Convert to NumPy
            image_np = np.array(image, dtype=np.uint8)  # (H, W, 3)

            return {
                "image": image_np,
                "height": image_np.shape[0],
                "width": image_np.shape[1],
                "save_path": save_path,
                "path": path
            }
        except Exception as e:
            print(f"Error opening {path}: {e}")
            return {
                "image": None,
                "height": 0,
                "width": 0,
                "save_path": save_path,
                "path": path
            }


def extract_largest_object_maskrcnn(
    image_np: np.ndarray, 
    model: torch.nn.Module, 
    device: str = "cuda", 
    score_threshold: float = 0.5
):
    """
    1. Convert image_np (H,W,3) to a Torch tensor, run Mask R-CNN.
    2. Filter detections with confidence >= score_threshold.
    3. If none pass, return None (indicating "no objects").
    4. Otherwise, pick the largest object mask (by area).
    5. Everything else is turned WHITE (255,255,255) in the image.
    """
    # Convert to Tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(Image.fromarray(image_np)).to(device)

    # Run inference
    with torch.no_grad():
        preds = model([image_tensor])  # list of dict, one per image
    pred = preds[0]  # only one image in the batch

    # Extract boxes, scores, and masks
    boxes = pred['boxes']       # [N, 4]
    scores = pred['scores']     # [N]
    masks = pred['masks']       # [N, 1, H, W]

    # Filter by confidence threshold
    keep = [i for i, s in enumerate(scores) if s >= score_threshold]

    if len(keep) == 0:
        # No objects above threshold
        return None

    masks = masks[keep]

    # Find largest object by mask area
    areas = []
    for i in range(masks.shape[0]):
        mask_i = (masks[i, 0] > 0.5)
        area = mask_i.sum().item()
        areas.append(area)

    largest_idx = max(range(len(areas)), key=lambda i: areas[i])
    # Binary mask for the largest object
    largest_mask = (masks[largest_idx, 0] > 0.5).cpu().numpy()

    # Make a copy to avoid modifying original
    masked_np = image_np.copy()
    # Turn everything else white
    masked_np[~largest_mask] = [255, 255, 255]

    return masked_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument('--dataset', type=str, default="cc")
    parser.add_argument('--split', type=str, choices=["train", "val"], default="val")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help="Confidence threshold for Mask R-CNN.")
    args = parser.parse_args()

    # Distributed setup (if you want to run multi-GPU)
    utils.init_distributed_mode(args)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    print(args)

    # Set seeds
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Basic checks
    assert args.dataset in IMAGE_ROOTS, f"Dataset {args.dataset} not found in data_path.py"

    # Enable GPU speed-ups
    cudnn.benchmark = True

    # 1) Load Mask R-CNN (TorchVision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    # 2) Create dataset / dataloader
    dataset = Image_Dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=False,
        seed=args.seed
    )
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=16,
        collate_fn=lambda x: x,  # trivial collate for dict
        drop_last=False,
    )

    # 3) Loop over images
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('empty_rate', utils.SmoothedValue(window_size=100, fmt='{value:.2f}'))
    header = f"Mask R-CNN largest object (white background) {args.dataset}/{args.split}"

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            # Filter out any images that failed to load
            tmp = [b for b in batch if b['image'] is not None]
            if len(tmp) == 0:
                # all images in this batch failed
                metric_logger.update(empty_rate=1.0)
                metric_logger.synchronize_between_processes()
                continue
            batch = tmp

            empty_count = 0
            for sample in batch:
                image_np = sample["image"]
                save_path = sample["save_path"]
                height, width = sample["height"], sample["width"]

                # 4) Extract largest object, or None if no objects
                masked_result = extract_largest_object_maskrcnn(
                    image_np, model, device, score_threshold=args.score_threshold
                )

                # 5) If no objects, save plain WHITE image
                if masked_result is None:
                    empty_count += 1
                    white_image = 255 * np.ones((height, width, 3), dtype=np.uint8)
                    out_pil = Image.fromarray(white_image)
                    out_pil.save(save_path)
                else:
                    # Otherwise, save the masked image (largest object on white background)
                    out_pil = Image.fromarray(masked_result)
                    out_pil.save(save_path)

            metric_logger.update(empty_rate=empty_count / len(batch))

            # Clean up
            del batch
            torch.cuda.empty_cache()
            metric_logger.synchronize_between_processes()

    print("Done.")
