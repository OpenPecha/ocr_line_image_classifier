from pathlib import Path

import pandas as pd

from ocr_line_image_classifier.update_transcript_sccore import (  # process_images_and_transcripts,
    calculate_similarity,
)

OUTPUT_DIR = Path("./tests/test_data/updated_transcript")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_similarity_score():
    expected_text = "Hello, World! How are you?"
    ocr_text = "Hello, World!"
    similarity_score = calculate_similarity(expected_text, ocr_text)
    rounded_similarity_score = round(similarity_score, 2)
    print(rounded_similarity_score)
    expected_score = 0.67
    assert rounded_similarity_score == expected_score


def test_process_images_and_transcripts():
    transcript_csv_path = Path("tests/test_data/transcript/transcript.csv")
    # image_dir = Path("tests/test_data/images")
    # process_images_and_transcripts(transcript_csv_path, image_dir)
    output_csv_path = OUTPUT_DIR / f"{transcript_csv_path.name}"
    output_df = pd.read_csv(output_csv_path)
    assert output_df.shape[0] == 8
    assert output_df.shape[1] == 7
