import argparse
import logging
import pathlib as path
from typing import cast

from lib.multimodal_search import verify_image_embedding

logger = logging.getLogger(__name__)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_img_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    _ = verify_img_parser.add_argument(
        "image", type=str, help="Path to image to be analysed"
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = path.Path(cast(str, args.image))
            logger.debug("image_path parsed: %s", image_path)
            verify_image_embedding(image_path)

        case _:
            parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
