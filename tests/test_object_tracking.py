import os
import pytest
import yaml
import json

from src.object_tracking import main
from src.definitions import CONFIG_PATH, ROOT_DIR

@pytest.fixture(scope="module")
def config_data():
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
        return config

@pytest.fixture(scope="module")
def meta_data(config_data):
    match = config_data["match"]
    meta_file_path = os.path.join(ROOT_DIR, str(match["meta_file"]))
    assert os.path.exists(meta_file_path), f"Meta file not found at {meta_file_path}"
    with open(meta_file_path, "r") as f:
        return json.load(f)

def test_config_is_valid(config_data):
    assert "match" in config_data
    assert "byte_track" in config_data
    assert "yolo" in config_data
    assert "output_jsonl_relative_file_path" in config_data

def test_meta_contains_essentials(meta_data):
    assert "fps" in meta_data
    assert "pitchLength" in meta_data
    assert "pitchWidth" in meta_data

def test_main_runs_successfully(tmp_path, config_data):
    """
    Run the main function and ensure it completes without error.
    """
    original_output_path = os.path.join(ROOT_DIR, str(config_data["output_jsonl_relative_file_path"]))
    if os.path.exists(original_output_path):
        os.rename(original_output_path, original_output_path + ".bak")
    try:
        result = main()
        assert result == 0
        assert os.path.exists(original_output_path), "Output JSONL file was not created."
    finally:
        if os.path.exists(original_output_path):
            os.remove(original_output_path)
        if os.path.exists(original_output_path + ".bak"):
            os.rename(original_output_path + ".bak", original_output_path)

def test_video_file_exists(config_data):
    video_path = os.path.join(ROOT_DIR, str(config_data["match"]["video"]))
    assert os.path.exists(video_path), f"Video file not found: {video_path}"