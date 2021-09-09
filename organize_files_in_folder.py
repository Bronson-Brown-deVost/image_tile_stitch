from pathlib import Path
import re
import shutil
from tqdm import tqdm
import logging

logging.basicConfig(
    filename=f"{Path(__file__).stem}.log",
    encoding="utf-8",
    level=logging.DEBUG,
    filemode="w",
)


p = re.compile("^(?P<prefix>.*)C0")

out_path = Path("/home/bronson/data/SQE/st_errors_organized")
if not out_path.exists():
    out_path.mkdir()

all_image_files = list(Path("/home/bronson/data/SQE/st_errors").rglob("*.jpg"))
all_images = [x.stem for x in all_image_files]
not_parsed = []
for image_file in tqdm(all_image_files):
    matches = p.search(image_file.stem)
    if matches:
        group_dict = matches.groupdict()
        if "prefix" in group_dict:
            prefix = group_dict["prefix"].rstrip("-")
            copy_dir = out_path / prefix
            if not copy_dir.exists():
                copy_dir.mkdir()
            logging.debug(f"moving {image_file.resolve()} to {copy_dir.resolve()}")
            shutil.move(image_file, copy_dir)
            logging.debug(f"moved {image_file.resolve()} to {copy_dir.resolve()}")
        else:
            not_parsed.append(image_file)
            logging.error(
                f"could not understand name {image_file.stem} of {image_file.resolve()}"
            )

print(len(not_parsed))
