import argparse
from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from tqdm.auto import tqdm

from idf.utils.common import instantiate_from_config, load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Test IDF on images")
    parser.add_argument(
        "--input_dir",
        default="assets/demo/noisy/",
        type=str,
        help="Directory of input images or path of a single image",
    )
    parser.add_argument(
        "--result_dir",
        default="assets/demo/denoised/",
        type=str,
        help="Directory to save denoised images",
    )
    parser.add_argument(
        "--config_dir",
        default="configs/models/idfnet.yaml",
        type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="pretrained_models/idf_g_15.ckpt",
        type=str,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--adaptive_iteration",
        action="store_true",
        help="Enable adaptive early stopping",
    )
    parser.add_argument(
        "--max_iteration",
        type=int,
        default=10,
        help="Maximum number of denoising iterations",
    )
    return parser.parse_args()


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }


def gather_files(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp] if is_image_file(inp) else []
    # Collect images in the top-level directory (non-recursive)
    files = [p for p in inp.iterdir() if p.is_file() and is_image_file(p)]
    return sorted(files)


def main():
    args = parse_args()

    inp_path = Path(args.input_dir)
    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = gather_files(inp_path)
    if len(files) == 0:
        raise FileNotFoundError(f"No image files found at {inp_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_config = OmegaConf.load(args.config_dir)
    model = instantiate_from_config(model_config)
    state_dict = torch.load(args.ckpt_dir, map_location="cpu")
    load_state_dict(model, state_dict, strict=True)
    model.freeze()
    model.eval()
    model.to(device)

    print(f"\n ==> Running with weights: {args.ckpt_dir}\n ")

    to_tensor = ToTensor()
    to_pil = ToPILImage()

    with torch.inference_mode():
        for file_ in tqdm(files, desc="Denoising", unit="img"):
            input_image = Image.open(file_).convert("RGB")
            img = to_tensor(input_image).unsqueeze(0).to(device)

            output = model(img, adaptive_iter=args.adaptive_iteration, max_iter=args.max_iteration)

            output_clamped = output.clamp(0.0, 1.0).squeeze(0).detach().to("cpu")
            output_image = to_pil(output_clamped)
            save_path = out_dir / f"{file_.stem}.png"
            output_image.save(save_path)

    print(f"\nDenoised images are saved at {out_dir}")


if __name__ == "__main__":
    main()