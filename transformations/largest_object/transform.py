import argparse
import glob
import numpy as np
import os
import random
import shutil
import sys
import time
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


def download_with_retry(file_path, local_path, retries=5, delay=5):
    """
    Downloads a file from Google Drive with retries in case of failure.
    """
    for attempt in range(retries):
        try:
            # Simulate the download logic (replace with actual API or copy logic)
            shutil.copy(file_path, local_path)  # Replace with actual download logic
            return True
        except Exception as e:
            print(f"Retry {attempt + 1}/{retries} failed: {e}")
            time.sleep(delay)
    print(f"Failed to download {file_path} after {retries} attempts.")
    return False


class Image_Dataset(Dataset):
    """
    Loads images from IMAGE_ROOTS[args.dataset]/args.split,
    downloads them in batches if necessary, and preprocesses them.
    """
    def __init__(self, args):
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

        self.preprocess = transforms.Resize(
            500, interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.local_cache = "local_cache"
        os.makedirs(self.local_cache, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        filename = os.path.basename(path)
        local_path = os.path.join(self.local_cache, filename)
        save_path = os.path.join(
            self.output_dir,
            os.path.splitext(filename)[0] + ".png"
        )

        # Ensure the file is cached locally
        if not os.path.exists(local_path):
            success = download_with_retry(path, local_path)
            if not success:
                print(f"Skipping {path} due to download failure.")
                return {
                    "image": None,
                    "height": 0,
                    "width": 0,
                    "save_path": save_path,
                    "path": path
                }

        try:
            # Load image from local cache
            with open(local_path, 'rb') as f:
                image = Image.open(f).convert("RGB")

            if min(image.size) > 500:
                image = self.preprocess(image)

            image_np = np.array(image, dtype=np.uint8)  # Convert to NumPy array

            return {
                "image": image_np,
                "height": image_np.shape[0],
                "width": image_np.shape[1],
                "save_path": save_path,
                "path": path
            }
        except Exception as e:
            print(f"Error processing {path}: {e}")
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
    Extracts the largest object in an image using Mask R-CNN.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(Image.fromarray(image_np)).to(device)

    with torch.no_grad():
        preds = model([image_tensor])
    pred = preds[0]

    boxes = pred['boxes']
    scores = pred['scores']
    masks = pred['masks']

    keep = [i for i, s in enumerate(scores) if s >= score_threshold]

    if len(keep) == 0:
        return None

    masks = masks[keep]

    areas = [(masks[i, 0] > 0.5).sum().item() for i in range(masks.shape[0])]
    largest_idx = max(range(len(areas)), key=lambda i: areas[i])
    largest_mask = (masks[largest_idx, 0] > 0.5).cpu().numpy()

    masked_np = image_np.copy()
    masked_np[~largest_mask] = [255, 255, 255]

    return masked_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument('--dataset', type=str, default="cc")
    parser.add_argument('--split', type=str, choices=["train", "val"], default="val")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--score_threshold', type=float, default=0.5)
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    dataset = Image_Dataset(args)
    sampler = DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False, seed=args.seed
    )
    data_loader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size,
        num_workers=4, collate_fn=lambda x: x, drop_last=False
    )

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('empty_rate', utils.SmoothedValue(window_size=100, fmt='{value:.2f}'))
    header = f"Mask R-CNN largest object (white background) {args.dataset}/{args.split}"

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            tmp = [b for b in batch if b['image'] is not None]
            if not tmp:
                metric_logger.update(empty_rate=1.0)
                metric_logger.synchronize_between_processes()
                continue
            batch = tmp

            empty_count = 0
            for sample in batch:
                image_np = sample["image"]
                save_path = sample["save_path"]
                height, width = sample["height"], sample["width"]

                masked_result = extract_largest_object_maskrcnn(
                    image_np, model, device, score_threshold=args.score_threshold
                )

                if masked_result is None:
                    empty_count += 1
                    white_image = 255 * np.ones((height, width, 3), dtype=np.uint8)
                    out_pil = Image.fromarray(white_image)
                    out_pil.save(save_path)
                else:
                    out_pil = Image.fromarray(masked_result)
                    out_pil.save(save_path)

            metric_logger.update(empty_rate=empty_count / len(batch))
            del batch
            torch.cuda.empty_cache()
            metric_logger.synchronize_between_processes()

    print("Done.")
