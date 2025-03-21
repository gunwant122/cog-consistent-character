from PIL import Image
from pathlib import Path
from typing import List, Union

IMAGE_FILE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
FORMAT_CHOICES = ["webp", "jpg", "png"]
DEFAULT_FORMAT = "webp"
DEFAULT_QUALITY = 80


def get_default_format() -> str:
    """Get default output format"""
    return DEFAULT_FORMAT


def get_default_quality() -> int:
    """Get default output quality"""
    return DEFAULT_QUALITY


def should_optimise_images(output_format: str, output_quality: int) -> bool:
    """Check if images should be optimized based on format and quality"""
    return output_quality < 100 or output_format in [
        "webp",
        "jpg",
    ]


def optimise_image_files(
    output_format: str = DEFAULT_FORMAT, 
    output_quality: int = DEFAULT_QUALITY, 
    files: List[Union[str, Path]] = []
) -> List[Path]:
    """Optimize image files with given format and quality"""
    if should_optimise_images(output_format, output_quality):
        optimised_files = []
        for file in files:
            file_path = Path(file)
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_FILE_EXTENSIONS:
                image = Image.open(file_path)
                optimised_file_path = file_path.with_suffix(f".{output_format}")
                image.save(
                    optimised_file_path,
                    quality=output_quality,
                    optimize=True,
                )
                optimised_files.append(optimised_file_path)
            else:
                optimised_files.append(file_path)

        return optimised_files
    else:
        return [Path(f) for f in files]
