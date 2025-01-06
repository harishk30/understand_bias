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


def file_path_generator(root, extensions, batch_size):
    """
    Generator to yield batches of file paths.
    """
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(root, f"*.{ext}")))

    for i in range(0, len(all_files), batch_size):
        yield all_files[i:i + batch_size]


class LazyImageDataset(Dataset):
    """
    Dataset with lazy loading for handling large directories.
    """
    def __init__(self, file_paths, output_dir):
        self.file_paths = file_paths
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.preprocess = transforms.Resize(
            500, interpolation=transforms.InterpolationMode.BICUBIC
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        save_path = os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(path))[0] + ".png"
        )

        try:
            with open(path, 'rb') as f:
                image = Image.open(f).convert("RGB")

            if min(image.size) > 500:
                image = self.preprocess(image)

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
            return None


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


def init_distributed_mode(args):
    """
    Initialize distributed mode for multi-GPU setup.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.distributed = False

    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=args.world_size, rank=args.rank
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument('--dataset', type=str, default="cc")
    parser.add_argument('--split', type=str, choices=["train", "val"], default="val")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--score_threshold', type=float, default=0.5)
    args = parser.parse_args()

    # Distributed setup (if needed)
    init_distributed_mode(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    root_dir = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
    output_dir = os.path.join(SAVE_ROOTS['largest_object'], args.dataset, args.split)
    extensions = ["jpg", "png", "JPEG"]
    chunk_size = 1000  # Adjust based on system capabilities

    for file_paths in file_path_generator(root_dir, extensions, chunk_size):
        dataset = LazyImageDataset(file_paths, output_dir)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            collate_fn=lambda x: [d for d in x if d],
            drop_last=False,
        )

        with torch.no_grad():
            for batch in data_loader:
                for sample in batch:
                    image_np = sample["image"]
                    save_path = sample["save_path"]
                    height, width = sample["height"], sample["width"]

                    masked_result = extract_largest_object_maskrcnn(
                        image_np, model, device, score_threshold=args.score_threshold
                    )

                    if masked_result is None:
                        white_image = 255 * np.ones((height, width, 3), dtype=np.uint8)
                        Image.fromarray(white_image).save(save_path)
                    else:
                        Image.fromarray(masked_result).save(save_path)

    print("Processing completed.")
