from pathlib import Path

import pandas as pd

from ocr_line_image_classifier.metrics import (
    classify_transcripts,
    evaluate_transcripts,
    save_evaluation_metrics,
)
from ocr_line_image_classifier.utils import (
    copy_images,
    create_output_directories,
    save_transcripts,
)


def group_images_by_similarity(
    updated_csv_path, image_dir, output_dir, similarity_threshold
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

    high_similarity_rows, low_similarity_rows = classify_transcripts(
        transcript_df, similarity_threshold
    )

    stats = []
    stats = evaluate_transcripts(transcript_df, similarity_threshold)
    output_file = (
        output_dir.parent / "stats" / f"evaluation_metrics_{similarity_threshold}.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_evaluation_metrics(stats, output_file)

    copy_images(high_similarity_rows, image_dir, high_similarity_dir)
    copy_images(low_similarity_rows, image_dir, low_similarity_dir)

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
    image_dir = Path("./data/line_image_with_issue/images")
    output_csv_path = Path(
        "./data/line_image_with_issue/updated_transcript/transcript.csv"
    )

    # Group images by similarity score and save in separate folders
    output_dir = Path("./data/line_image_with_issue/filtered_images")
    group_images_by_similarity(
        output_csv_path, image_dir, output_dir, similarity_threshold=0.95
    )
