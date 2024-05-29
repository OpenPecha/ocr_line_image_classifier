import json
from pathlib import Path

import pandas as pd

from ocr_line_image_classifier.metrics import (
    classify_transcripts,
    evaluate_transcripts,
    save_evaluation_metrics,
)
from ocr_line_image_classifier.utils import copy_images, create_output_directories


def copy_images_and_save_transcript(image_dir, transcript_rows, output_dir):
    """Copy images to the target directory and save transcript."""
    transcript_output_dir = output_dir / "transcript"

    copy_images(transcript_rows, image_dir, output_dir)

    transcript_file = transcript_output_dir / "transcript.csv"
    transcript_rows.to_csv(
        transcript_file, mode="a", header=not transcript_file.exists(), index=False
    )
    transcript_file_json = transcript_output_dir / "transcript.json"
    if transcript_file_json.exists():
        with open(transcript_file_json, encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_df = pd.DataFrame(existing_data)
        updated_df = pd.concat([existing_df, transcript_rows], ignore_index=True)
    else:
        updated_df = transcript_rows

    updated_df.to_json(
        transcript_file_json, orient="records", force_ascii=False, indent=4
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

        stats = []
        stats = evaluate_transcripts(transcript_df, similarity_threshold)
        evaluation_file = (
            output_folder.parent
            / "stats"
            / batch_name
            / f"{batch_name}_evaluation_{similarity_threshold}.csv"
        )
        evaluation_file.parent.mkdir(parents=True, exist_ok=True)
        save_evaluation_metrics(stats, evaluation_file)

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
    transcript_folder = Path("./data/norbuketaka/updated_transcript")
    output_folder = Path("./data/norbuketaka/filtered_images")
    similarity_threshold = 0.95
    process_updated_batches(
        image_folder, transcript_folder, output_folder, similarity_threshold
    )
