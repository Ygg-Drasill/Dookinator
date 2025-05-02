import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def extract_positions(frame):
    # Combine home and away player XY positions
    positions = []
    for player in frame.get("homePlayers", []) + frame.get("awayPlayers", []):
        xyz = player["xyz"]
        positions.append(xyz[:2])  # just x and y
    return np.array(positions)


def compute_frame_similarity(points1, points2):
    if len(points1) == 0 or len(points2) == 0:
        return np.inf  # no data to compare

    # Pad smaller array with dummy points to make them equal in length
    max_len = max(len(points1), len(points2))
    pad1 = np.pad(points1, ((0, max_len - len(points1)), (0, 0)), mode='constant', constant_values=0)
    pad2 = np.pad(points2, ((0, max_len - len(points2)), (0, 0)), mode='constant', constant_values=0)

    # Create cost matrix (Euclidean distance)
    cost_matrix = np.linalg.norm(pad1[:, None, :] - pad2[None, :, :], axis=2)

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    # Return average distance per point
    return total_cost / max_len


def compare_files(file1_path, file2_path):
    total_score = 0
    frame_count = 0

    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        for line_f1, line_f2 in tqdm(zip(f1, f2), total=750):
            frame_f1 = json.loads(line_f1)
            frame_f2 = json.loads(line_f2)

            points1 = extract_positions(frame_f1)
            points2 = extract_positions(frame_f2)

            similarity_score = compute_frame_similarity(points1, points2)
            total_score += similarity_score
            frame_count += 1

    average_score = total_score / frame_count
    return average_score
def main():
    file1 = "comparison_data/file1"
    file2 = "comparison_data/file2"

    score = compare_files(file1, file2)
    print(score)