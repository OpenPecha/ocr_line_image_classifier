# ocr_line_image_classifier/shared_utils.py
import shutil


def copy_images(rows, image_dir, target_dir):
    """Copy images to the target directory."""
    for _, row in rows.iterrows():
        image_file = image_dir / row["line_image_id"]
        if image_file.exists():
            shutil.copy(image_file, target_dir / "images" / image_file.name)


def save_transcript(transcript_df, output_csv_path, output_json_path):
    """Save the updated transcript DataFrame to a CSV file."""
    transcript_df.to_csv(output_csv_path, index=False)
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(transcript_df.to_json(orient="records", force_ascii=False, indent=4))


def create_output_directories(output_dir):
    """Create output directories for high and low similarity images."""
    high_similarity_dir = output_dir / "valid_data_without_issue"
    low_similarity_dir = output_dir / "valid_data_with_issue"
    (high_similarity_dir / "images").mkdir(parents=True, exist_ok=True)
    (low_similarity_dir / "images").mkdir(parents=True, exist_ok=True)
    (high_similarity_dir / "transcript").mkdir(parents=True, exist_ok=True)
    (low_similarity_dir / "transcript").mkdir(parents=True, exist_ok=True)
    return high_similarity_dir, low_similarity_dir


def save_transcripts(
    high_similarity_rows, low_similarity_rows, high_similarity_dir, low_similarity_dir
):
    """Save new transcript CSV files for high and low similarity groups."""
    save_transcript(
        high_similarity_rows,
        high_similarity_dir / "transcript" / "transcript.csv",
        high_similarity_dir / "transcript" / "transcript.json",
    )
    save_transcript(
        low_similarity_rows,
        low_similarity_dir / "transcript" / "transcript.csv",
        low_similarity_dir / "transcript" / "transcript.json",
    )
