import shutil
from pathlib import Path

import pandas as pd


def create_output_directories(output_dir):
    """Create output directories for valid data with and without issues."""
    valid_data_without_issue_dir = output_dir / "valid_data_without_issue"
    valid_data_with_issue_dir = output_dir / "valid_data_with_issue"

    (valid_data_without_issue_dir / "images").mkdir(parents=True, exist_ok=True)
    (valid_data_without_issue_dir / "transcript").mkdir(parents=True, exist_ok=True)
    (valid_data_with_issue_dir / "images").mkdir(parents=True, exist_ok=True)
    (valid_data_with_issue_dir / "transcript").mkdir(parents=True, exist_ok=True)

    return valid_data_without_issue_dir, valid_data_with_issue_dir


def classify_transcripts(transcript_df, similarity_threshold):
    """Classify transcript rows based on similarity score."""
    high_similarity_rows = transcript_df[
        transcript_df["similarity_score"] >= similarity_threshold
    ]
    low_similarity_rows = transcript_df[
        transcript_df["similarity_score"] < similarity_threshold
    ]
    return high_similarity_rows, low_similarity_rows


def copy_images_and_save_transcript(image_dir, transcript_rows, output_dir):
    """Copy images to the target directory and save transcript."""
    image_output_dir = output_dir / "images"
    transcript_output_dir = output_dir / "transcript"

    for index, row in transcript_rows.iterrows():
        image_file = image_dir / row["line_image_id"]
        if image_file.exists():
            shutil.copy(image_file, image_output_dir)

    transcript_file = transcript_output_dir / "transcript.csv"
    transcript_rows.to_csv(
        transcript_file, mode="a", header=not transcript_file.exists(), index=False
    )


def process_updated_batches(
    image_folder, transcript_folder, output_folder, similarity_threshold=0.9
):
    """Process updated batch transcripts and images."""
    valid_data_without_issue_dir, valid_data_with_issue_dir = create_output_directories(
        output_folder
    )

    for transcript_file in transcript_folder.glob("*.csv"):
        transcript_df = pd.read_csv(transcript_file)
        batch_name = transcript_file.stem

        high_similarity_rows, low_similarity_rows = classify_transcripts(
            transcript_df, similarity_threshold
        )

        if not high_similarity_rows.empty:
            copy_images_and_save_transcript(
                image_folder / batch_name,
                high_similarity_rows,
                valid_data_without_issue_dir,
            )
        if not low_similarity_rows.empty:
            copy_images_and_save_transcript(
                image_folder / batch_name,
                low_similarity_rows,
                valid_data_with_issue_dir,
            )


if __name__ == "__main__":
    image_folder = Path("./data/norbuketaka/images")
    transcript_folder = Path("./data/norbuketaka/updated_transcripts")
    output_folder = Path("./data/norbuketaka/filtered_images")
    similarity_threshold = 0.85
    process_updated_batches(
        image_folder, transcript_folder, output_folder, similarity_threshold
    )
