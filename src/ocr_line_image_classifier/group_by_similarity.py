import shutil
from pathlib import Path

import pandas as pd


def save_updated_transcript(transcript_df, output_csv_path):
    """Save the updated transcript DataFrame to a CSV file."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_df.to_csv(output_csv_path, index=False)
    print(
        f"OCR and similarity calculation completed. Updated CSV saved as '{output_csv_path}'."
    )


def create_output_directories(output_dir):
    """Create output directories for high and low similarity images."""
    high_similarity_dir = output_dir / "image_without_issue"
    low_similarity_dir = output_dir / "image_with_issue"
    high_similarity_dir.mkdir(parents=True, exist_ok=True)
    low_similarity_dir.mkdir(parents=True, exist_ok=True)
    return high_similarity_dir, low_similarity_dir


def classify_images(transcript_df, image_dir, similarity_threshold):
    """Classify images based on similarity score."""
    high_similarity_rows = []
    low_similarity_rows = []

    for _, row in transcript_df.iterrows():
        similarity_score = row["similarity_score"]

        if similarity_score >= similarity_threshold:
            high_similarity_rows.append(row)
        else:
            low_similarity_rows.append(row)

    return high_similarity_rows, low_similarity_rows


def copy_images(rows, image_dir, target_dir):
    """Copy images to the target directory."""
    for row in rows:
        image_file = image_dir / row["line_image_id"]
        if image_file.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_file, target_dir)


def save_transcripts(
    high_similarity_rows, low_similarity_rows, high_similarity_dir, low_similarity_dir
):
    """Save new transcript CSV files for high and low similarity groups."""
    high_similarity_df = pd.DataFrame(high_similarity_rows)
    low_similarity_df = pd.DataFrame(low_similarity_rows)

    save_updated_transcript(
        high_similarity_df, high_similarity_dir / "transcript" / "transcript.csv"
    )
    save_updated_transcript(
        low_similarity_df, low_similarity_dir / "transcript" / "transcript.csv"
    )


def group_images_by_similarity(
    updated_csv_path, image_dir, output_dir, similarity_threshold=0.9
):
    """
    Group images by similarity score and save them in separate folders.

    Parameters:
    - updated_csv_path (str): Path to the updated CSV file.
    - image_dir (str): Directory where the line images are stored.
    - output_dir (str): Directory to save the grouped images and transcripts.
    - similarity_threshold (float): Threshold to separate images.
    """
    transcript_df = pd.read_csv(updated_csv_path)

    high_similarity_dir, low_similarity_dir = create_output_directories(output_dir)

    high_similarity_rows, low_similarity_rows = classify_images(
        transcript_df, image_dir, similarity_threshold
    )

    copy_images(high_similarity_rows, image_dir, high_similarity_dir / "images")
    copy_images(low_similarity_rows, image_dir, low_similarity_dir / "images")

    save_transcripts(
        high_similarity_rows,
        low_similarity_rows,
        high_similarity_dir,
        low_similarity_dir,
    )

    print(
        f"Images and transcripts grouped by similarity threshold {similarity_threshold}."
    )


if __name__ == "__main__":
    transcript_csv_path = Path("./data/line_image_with_issue/transcript/transcript.csv")
    image_dir = Path("./data/line_image_with_issue/images")
    output_csv_path = Path(
        "./data/line_image_with_issue/transcript/updated_transcript.csv"
    )

    # Group images by similarity score and save in separate folders
    output_dir = Path("./data/line_image_with_issue/grouped_images")
    group_images_by_similarity(
        output_csv_path, image_dir, output_dir, similarity_threshold=0.99
    )
