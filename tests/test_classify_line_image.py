import shutil
from pathlib import Path

import pandas as pd

from ocr_line_image_classifier.classify_issue import group_images_by_similarity


# Test for group_images_by_similarity
def test_group_images_by_similarity():
    image_dir = Path("tests/test_data/images")
    output_dir = Path("tests/test_data/filtered_images")

    updated_csv_path = Path("tests/test_data/updated_transcript/transcript.csv")

    # Clear previous test results if any
    if output_dir.exists():
        shutil.rmtree(output_dir)

    group_images_by_similarity(
        updated_csv_path, image_dir, output_dir, similarity_threshold=0.90
    )

    high_similarity_dir = output_dir / "valid_data_without_issue"
    low_similarity_dir = output_dir / "valid_data_with_issue"

    high_similarity_csv = high_similarity_dir / "transcript" / "transcript.csv"
    low_similarity_csv = low_similarity_dir / "transcript" / "transcript.csv"

    # Ensure directories and files are created
    assert (high_similarity_dir / "images").exists()
    assert (low_similarity_dir / "images").exists()
    assert high_similarity_csv.exists()
    assert low_similarity_csv.exists()

    # Load the expected data from CSV files
    expected_high_similarity_df = pd.read_csv(
        "tests/test_data/expected_transcript/high_simi_transcript.csv"
    )
    expected_low_similarity_df = pd.read_csv(
        "tests/test_data/expected_transcript/low_simi_transcript.csv"
    )

    # Load the actual results
    high_transcript_df = pd.read_csv(high_similarity_csv)
    low_transcript_df = pd.read_csv(low_similarity_csv)

    # Verify the high similarity transcript data
    pd.testing.assert_frame_equal(high_transcript_df, expected_high_similarity_df)

    # Verify the low similarity transcript data
    pd.testing.assert_frame_equal(low_transcript_df, expected_low_similarity_df)
