import cv2
import Levenshtein
import pandas as pd
from line_image_to_text.ocr_line_image_and_rearrange_json import ocr_process_image


def calculate_similarity(text1, text2):
    """Calculate the Levenshtein similarity score between two texts."""
    return Levenshtein.ratio(text1, text2)


def process_images_and_transcripts(transcript_csv_path, image_dir, output_csv_path):
    """
    Process images and transcripts to perform OCR, calculate similarity scores, and save the results.

    Parameters:
    - transcript_csv_path (str): Path to the transcript CSV file.
    - image_dir (str): Directory where the line images are stored.
    - output_csv_path (str): Path to save the updated CSV file with OCR results and similarity scores.
    """
    # Read the transcript CSV file
    transcript_df = pd.read_csv(transcript_csv_path)

    # Initialize lists to store OCR results and similarity scores
    ocr_texts = []
    similarity_scores = []

    # Iterate over each row in the DataFrame
    for index, row in transcript_df.iterrows():
        # Get the image file path and expected text
        image_file = f"{image_dir}/{row['line_image_id']}"
        expected_text = row["text"]

        # Load the image
        image = cv2.imread(image_file)

        if image is None:
            print(f"Error loading image: {image_file}")
            ocr_texts.append("")
            similarity_scores.append(0)
            continue

        # Perform OCR on the image
        ocr_text = ocr_process_image(image_file)

        # Calculate the similarity score
        similarity_score = calculate_similarity(expected_text, ocr_text)

        # Append the results to the lists
        ocr_texts.append(ocr_text)
        similarity_scores.append(similarity_score)

    # Add the OCR text and similarity score to the DataFrame
    transcript_df["ocr_text"] = ocr_texts
    transcript_df["similarity_score"] = similarity_scores

    # Save the updated DataFrame to a new CSV file
    transcript_df.to_csv(output_csv_path, index=False)

    print(
        f"OCR and similarity calculation completed. Updated CSV saved as '{output_csv_path}'."
    )


# Example usage
transcript_csv_path = "./data/line_image_with_issue/transcript/transcript.csv"
image_dir = "./data/line_image_with_issue/images"
output_csv_path = "./data/line_image_with_issue/transcript/updated_transcript.csv"

process_images_and_transcripts(transcript_csv_path, image_dir, output_csv_path)
