from multiprocessing import Pool
from pathlib import Path

import cv2
import Levenshtein
import pandas as pd
from line_image_to_text.ocr_line_image_and_rearrange_json import ocr_process_image
from tqdm import tqdm

from ocr_line_image_classifier.checkpoint import (
    load_checkpoints,
    save_checkpoint,
    save_corrupted_files,
)

OUTPUT_DIR = Path("./tests/test_data/updated_transcript")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def save_transcript(transcript_df, output_csv_path):
    """Save the updated transcript DataFrame to a CSV file."""
    transcript_df.to_csv(output_csv_path, index=False)
    print(
        f"OCR and similarity calculation completed. Updated CSV saved as '{output_csv_path}'."
    )


def update_transcript_dataframe(transcript_df, image_dir):
    """
    Update the transcript DataFrame with OCR results and similarity scores.

    Parameters:
    - transcript_df (pd.DataFrame): DataFrame containing the transcript information.
    - image_dir (Path): Directory where the line images are stored.

    Returns:
    - pd.DataFrame: Updated DataFrame with OCR results and similarity scores.
    """
    ocr_texts = []
    similarity_scores = []

    for index, row in tqdm(
        transcript_df.iterrows(), total=transcript_df.shape[0], desc="Processing images"
    ):
        image_file = image_dir / row["line_image_id"]
        expected_text = row["text"]

        image = load_image(image_file)

        if image is not None:
            ocr_text = ocr_process_image(image_file)
            similarity_score = calculate_similarity(expected_text, ocr_text)
        else:
            ocr_text = ""
            similarity_score = 0

        ocr_texts.append(ocr_text)
        similarity_scores.append(similarity_score)

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
    save_transcript(updated_transcript_df, output_csv_path)


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


if __name__ == "__main__":

    image_folder = "./data/norbuketaka/images"
    transcript_folder = "./data/norbuketaka/transcript"
    num_processes = 4
    results = process_batches_parallel(image_folder, transcript_folder, num_processes)
