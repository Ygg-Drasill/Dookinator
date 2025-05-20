from typing import Generator, Iterable, List, TypeVar, Iterator

import numpy as np
import supervision as sv
import torch
import umap
import yaml
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from supervision import Detections
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

from src.definitions import CONFIG_PATH

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    yolo = config['yolo']


V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'




def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:

    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch

def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

class TeamClassifier:

    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = DBSCAN(min_samples=2)
        self.knn_classifier = None  # Will be initialized in fit

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        labels = self.cluster_model.fit_predict(projections)

        mask = labels != -1
        self.knn_classifier = KNeighborsClassifier(n_neighbors=1)
        self.knn_classifier.fit(projections[mask], labels[mask])

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if self.knn_classifier is None:
            raise RuntimeError("Model has not been fitted yet.")

        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.knn_classifier.predict(projections)

device = "cuda" if torch.cuda.is_available() else "cpu"
team_classifier = TeamClassifier(device=device)

def player_separation(frames, detections_chunk):
    crops = []
    PLAYER_CLASS_ID = yolo['selected_class_ids']['player']['id']
    GOALKEEPER_CLASS_ID = yolo['selected_class_ids']['goalkeeper']['id']

    color_lookups = []

    for i, frame in enumerate(frames):

        if i == 0:
            sorted_indices = np.argsort(detections_chunk[i].xyxy[:, 0])  # sort by x_min
            sorted_detections = detections_chunk[i][sorted_indices]

            detections_chunk[i] = sorted_detections

            pass

        crops += get_crops(frame, detections_chunk[i][detections_chunk[i].class_id == PLAYER_CLASS_ID])


    team_classifier.fit(crops)

    for i, frame in enumerate(frames):

        players = detections_chunk[i][detections_chunk[i].class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections_chunk[i][detections_chunk[i].class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        detections_chunk[i] = sv.Detections.merge([players, goalkeepers])

        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist()
        )
        color_lookups.append(color_lookup)
        detections_chunk[i]["player_team"] = color_lookups[i]

    return detections_chunk

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]