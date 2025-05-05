import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.definitions import ROOT_DIR


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

def extract_team_positions(frame):
    # Separate home and away team player XY positions
    home = [player["xyz"][:2] for player in frame.get("homePlayers", [])]
    away = [player["xyz"][:2] for player in frame.get("awayPlayers", [])]
    return np.array(home), np.array(away)

def plot_side_by_side(frame1, frame2, title1, title2, supertitle):
    home1, away1 = extract_team_positions(frame1)
    home2, away2 = extract_team_positions(frame2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(supertitle, fontsize=16)

    # Dataset 1
    axes[0].scatter(*home1.T, c='blue', label='Home', s=60)
    axes[0].scatter(*away1.T, c='red', label='Away', s=60)
    axes[0].set_title(title1)
    axes[0].set_xlim(-60, 60)
    axes[0].set_ylim(-40, 40)
    axes[0].set_aspect('equal', 'box')
    axes[0].grid(True)
    axes[0].legend()

    # Dataset 2
    axes[1].scatter(*home2.T, c='blue', label='Home', s=60)
    axes[1].scatter(*away2.T, c='red', label='Away', s=60)
    axes[1].set_title(title2)
    axes[1].set_xlim(-60, 60)
    axes[1].set_ylim(-40, 40)
    axes[1].set_aspect('equal', 'box')
    axes[1].grid(True)
    axes[1].legend()

    plt.show()

def plot_first_and_last_side_by_side(file1_path, file2_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    first_frame_1 = json.loads(lines1[0])
    last_frame_1 = json.loads(lines1[-1])
    first_frame_2 = json.loads(lines2[0])
    last_frame_2 = json.loads(lines2[-1])

    plot_side_by_side(first_frame_1, first_frame_2,
                      "Dataset 1 - First Frame", "Dataset 2 - First Frame",
                      "First Frame Comparison")

    plot_side_by_side(last_frame_1, last_frame_2,
                      "Dataset 1 - Last Frame", "Dataset 2 - Last Frame",
                      "Last Frame Comparison")


def main():

    file1 = os.path.join(ROOT_DIR, 'tests', 'jsonl_comparison', 'comparison_data', 'file1.jsonl')
    file2 = os.path.join(ROOT_DIR, 'tests', 'jsonl_comparison', 'comparison_data', 'file2.jsonl')


    score = compare_files(file1, file2)
    plot_first_and_last_side_by_side(file1, file2)
    print(score)

if __name__ == '__main__':
    sys.exit(main())