import os
import io
import logging
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor


def process_directory(dir_code: str, input_dir: str, output_dir: str):
    logging.info(f"Processing {dir_code}")
    # Get list of subdirectories
    zipfile_list = [entry for entry in os.scandir(f"{input_dir}/output_{dir_code}") if entry.name.endswith(".zip")]

    logging.debug(zipfile_list)


    with ZipFile(f"{output_dir}/{dir_code}_image.zip", "w") as output_image_zip:
        for zip_entry in zipfile_list:
            logging.debug(zip_entry.name)
            with ZipFile(zip_entry.path, "r") as inzip:
                for file in inzip.filelist:
                    try:
                        new_filename = "{}.{}".format(".".join(file.filename.split(".")[:-1]), "jpg")
                        img_file = inzip.read(file.filename)
                        img = Image.open(io.BytesIO(img_file)).convert("RGB")
                        output_bytes = io.BytesIO()
                        img.save(output_bytes, format="jpeg")
                        output_image_zip.writestr(new_filename, output_bytes.getvalue())
                    except:
                        logging.error(f"Error with file : {file.filename}")

    return



def main():

    # Define output directory
    output_dir = "/pl/active/acuna/image-caption-pair/meta-scir/meta-image-all"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define input root directory
    input_dir = "/pl/active/acuna/image-caption-pair/meta-scir/stage00_image_caption_meta"

    # Define output variations
    first_level_list = [f"{l1}{l2}" for l1 in "0123456789abcdef" for l2 in "0123456789abcdef"]

    #logging.basicConfig(filename=output_path / "log.out", level=logging.INFO, format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    with ProcessPoolExecutor(max_workers=32) as executor:
        [executor.submit(process_directory,
            first_level_dir,
            input_dir,
            output_dir) for first_level_dir in first_level_list]


if __name__ == "__main__":
    main()