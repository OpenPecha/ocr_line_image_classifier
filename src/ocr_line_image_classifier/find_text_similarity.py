import cv2
import Levenshtein
import pandas as pd
from line_image_to_text.ocr_line_image_and_rearrange_json import ocr_process_image
from tqdm import tqdm


def calculate_similarity(text1, text2):
    """Calculate the Levenshtein similarity score between two texts."""
    return Levenshtein.ratio(text1, text2)


def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
    return image


def update_transcript_dataframe(transcript_df, image_dir):
    """
    Update the transcript DataFrame with OCR results and similarity scores.

    Parameters:
    - transcript_df (pd.DataFrame): DataFrame containing the transcript information.
    - image_dir (str): Directory where the line images are stored.

    Returns:
    - pd.DataFrame: Updated DataFrame with OCR results and similarity scores.
    """
    ocr_texts = []
    similarity_scores = []

    for index, row in tqdm(
        transcript_df.iterrows(), total=transcript_df.shape[0], desc="Processing images"
    ):
        image_file = f"{image_dir}/{row['line_image_id']}"
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


def save_updated_transcript(transcript_df, output_csv_path):
    """Save the updated transcript DataFrame to a CSV file."""
    transcript_df.to_csv(output_csv_path, index=False)
    print(
        f"OCR and similarity calculation completed. Updated CSV saved as '{output_csv_path}'."
    )


def process_images_and_transcripts(transcript_csv_path, image_dir, output_csv_path):
    """
    Process images and transcripts to perform OCR, calculate similarity scores, and save the results.

    Parameters:
    - transcript_csv_path (str): Path to the transcript CSV file.
    - image_dir (str): Directory where the line images are stored.
    - output_csv_path (str): Path to save the updated CSV file with OCR results and similarity scores.
    """
    transcript_df = pd.read_csv(transcript_csv_path)
    updated_transcript_df = update_transcript_dataframe(transcript_df, image_dir)
    save_updated_transcript(updated_transcript_df, output_csv_path)


if __name__ == "__main__":
    transcript_csv_path = "./data/line_image_with_issue/transcript/transcript.csv"
    image_dir = "./data/line_image_with_issue/images"
    output_csv_path = "./data/line_image_with_issue/transcript/updated_transcript.csv"

    process_images_and_transcripts(transcript_csv_path, image_dir, output_csv_path)
