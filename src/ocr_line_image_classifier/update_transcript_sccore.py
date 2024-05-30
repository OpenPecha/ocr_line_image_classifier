from multiprocessing import Pool
from pathlib import Path

import cv2
import Levenshtein
import numpy as np
import pandas as pd
import pytesseract
from line_image_to_text.ocr_line_image_and_rearrange_json import ocr_process_image
from PIL import Image
from tqdm import tqdm

from ocr_line_image_classifier.checkpoint import (
    load_checkpoints,
    save_checkpoint,
    save_corrupted_files,
)
from ocr_line_image_classifier.utils import save_transcript

OUTPUT_DIR = Path("./tests/test_data/updated_transcript")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(image_path):
    """Preprocess the image for better OCR accuracy."""
    # Load the image using OpenCV
    image = cv2.imread(str(image_path))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to improve DPI
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

    # Apply median blurring to reduce noise
    blurred = cv2.medianBlur(gray, 3)

    # Apply adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Sharpen image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(adaptive_threshold, -1, kernel)

    return sharpened


def get_ocr_text(image_path):

    preprocessed_image = preprocess_image(image_path)

    # Save the preprocessed image with correct DPI using PIL
    binary_pil = Image.fromarray(preprocessed_image)
    temp_image_path = f"temp_{image_path.name}"
    binary_pil.save(temp_image_path, dpi=(300, 300))
    binary_pil_with_dpi = Image.open(temp_image_path)

    # Use pytesseract to extract text directly from the PIL image with correct DPI
    custom_oem_psm_config = r"--oem 3 --psm 4"
    text = pytesseract.image_to_string(
        binary_pil_with_dpi, lang="bod", config=custom_oem_psm_config
    )
    Path(temp_image_path).unlink()  # Delete the temporary image file

    return text.strip()

    # Optionally, delete the temporary image file


def calculate_similarity(text1, text2):
    """Calculate the Levenshtein similarity score between two texts."""
    return Levenshtein.ratio(text1, text2)


def load_image(image_path):
    """Load an image from a file path."""
    try:
        image = cv2.imread(str(image_path))  # Convert Path object to string
        if image is None:
            print(f"Error loading image: {image_path}")
            save_corrupted_files(image_path, "Error loading image")
        return image
    except Exception as e:
        print(f"Error loading image: {image_path}")
        save_corrupted_files(image_path, f"Error loading image: {e}")
        return None


def update_transcript_dataframe(transcript_df, image_dir):
    """
    Update the transcript DataFrame with OCR results and similarity scores.

    Parameters:
    - transcript_df (pd.DataFrame): DataFrame containing the transcript information.
    - image_dir (Path): Directory where the line images are stored.

    Returns:
    - pd.DataFrame: Updated DataFrame with OCR results and similarity scores.
    """
    pre_processsed_ocr_texts = []
    ocr_texts = []
    similarity_scores = []

    for index, row in tqdm(
        transcript_df.iterrows(), total=transcript_df.shape[0], desc="Processing images"
    ):
        image_file = image_dir / row["line_image_id"]
        expected_text = row["text"]

        image = load_image(image_file)

        if image is not None:
            pre_processsed_ocr_text = get_ocr_text(image_file)
            ocr_text = ocr_process_image(image_file)
            similarity_score = calculate_similarity(expected_text, ocr_text)

        else:
            pre_processsed_ocr_text = ""
            ocr_text = ""
            similarity_score = 0

        pre_processsed_ocr_texts.append(pre_processsed_ocr_text)
        ocr_texts.append(ocr_text)
        similarity_scores.append(similarity_score)

    transcript_df["pre_processed_ocr_text"] = pre_processsed_ocr_texts
    transcript_df["ocr_text"] = ocr_texts
    transcript_df["similarity_score"] = similarity_scores

    return transcript_df


def process_images_and_transcripts(transcript_csv_path, image_dir):
    """
    Process images and transcripts to perform OCR, calculate similarity scores, and save the results.

    Parameters:
    - transcript_csv_path (Path): Path to the transcript CSV file.
    - image_dir (Path): Directory where the line images are stored.
    """
    transcript_df = pd.read_csv(transcript_csv_path)
    updated_transcript_df = update_transcript_dataframe(transcript_df, image_dir)
    output_csv_path = OUTPUT_DIR / f"{transcript_csv_path.name}"
    output_json_path = OUTPUT_DIR / f"{transcript_csv_path.stem}.json"
    save_transcript(updated_transcript_df, output_csv_path, output_json_path)


def process_batch(batch_info):
    transcript_csv_path, image_dir = batch_info
    batch_name = image_dir.name
    checkpoints = load_checkpoints()
    if batch_name in checkpoints:
        print(f"Skipping batch: {batch_name}")
        return  # Skip processing if batch is in checkpoints

    process_images_and_transcripts(transcript_csv_path, image_dir)
    save_checkpoint(batch_name)


def process_batches_parallel(image_folder, transcript_folder, num_processes=4):
    """Process each batch in the image folder using multiprocessing."""
    image_folder_path = Path(image_folder)
    transcript_folder_path = Path(transcript_folder)

    batch_info_list = []
    for batch_folder in image_folder_path.iterdir():
        if batch_folder.is_dir():
            batch_transcript_csv = transcript_folder_path / f"{batch_folder.name}.csv"
            if batch_transcript_csv.exists():
                batch_info_list.append((batch_transcript_csv, batch_folder))

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_batch, batch_info_list),
                total=len(batch_info_list),
                desc="Processing batches",
            )
        )

    return results


"""if __name__ == "__main__":

    image_folder = "./data/norbuketaka/images"
    transcript_folder = "./data/norbuketaka/transcript"
    num_processes = 4
    results = process_batches_parallel(image_folder, transcript_folder, num_processes)
"""
if __name__ == "__main__":
    image_folder = Path("./data/line_image_with_issue/images")
    transcript_csv = Path("./data/line_image_with_issue/transcript/transcript.csv")
    process_images_and_transcripts(transcript_csv, image_folder)
