"""
计算变化检测的指标
1.pred_dir和label_dir中对应的文件其文件名应一致
2.输入都应是mask的图片
"""

import argparse
import os
import cv2
import numpy as np
from tqdm import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="/data5/tangyinzhou/LISA_xBD/pred",
        help="path of the prediction results",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default="/data5/tangyinzhou/LISA_xBD/label",
        help="path of the label",
    )
    parser.add_argument(
        "--no_change_sample",
        type=str,
        default="set_one_IoU",
        choices=["regular", "ignore", "set_one_IoU"],
        help="How to treat non-change sample: regular for no treatment, ignore for ignoring these sample when computing metrics, set_one for set the IoU for these samples as one",
    )
    args = parser.parse_args()
    return args


def cal_metrics(pred_path, label_path, no_change_sample):
    try:
        mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    except:
        raise ValueError("mask or label img error")
    # 确保 mask 和 label 是二值图像（0 和 1）
    mask = (mask > 0).astype(np.uint8)
    label = (label > 0).astype(np.uint8)
    if no_change_sample == "ignore" and np.sum(label == 1) == 0:
        return None
    # 计算 TP, FP, TN, FN
    TP = np.sum((mask == 1) & (label == 1))
    FP = np.sum((mask == 1) & (label == 0))
    TN = np.sum((mask == 0) & (label == 0))
    FN = np.sum((mask == 0) & (label == 1))

    # 计算指标
    OA = (TP + TN) / (TP + TN + FP + FN)
    Prec = TP / (TP + FP) if (TP + FP) != 0 else 0
    Rec = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = 2 * (Prec * Rec) / (Prec + Rec) if (Prec + Rec) != 0 else 0
    IoU = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
    if no_change_sample == "set_one_IoU" and np.sum(label == 1) == 0:
        IoU = 1
    # 计算 Kappa Coefficient (KC)
    P = ((TP + FP) * (TP + FN) + (FN + TN) * (TN + FP)) / (TP + TN + FP + FN) ** 2
    KC = (OA - P) / (1 - P) if (1 - P) != 0 else 0

    # 计算 False Alarm (FA)
    FA = FP / (TN + FP) if (TN + FP) != 0 else 0
    return {
        "OA": OA,
        "Prec": Prec,
        "Rec": Rec,
        "F1": F1,
        "IoU": IoU,
        "KC": KC,
        "FA": FA,
    }


def main():
    args = parse_args()
    images = [x for x in os.listdir(args.pred_dir)]
    overall_metrics = dict()

    for img in tqdm(images):
        pred_path = os.path.join(args.pred_dir, img)
        label_path = os.path.join(args.label_dir, img)
        if os.path.exists(pred_path) and os.path.exists(label_path):
            metrics = cal_metrics(
                pred_path=pred_path,
                label_path=label_path,
                no_change_sample=args.no_change_sample,
            )
            if not metrics:
                continue
            for metric, value in metrics.items():
                if not metric in overall_metrics.keys():
                    overall_metrics[metric] = []
                overall_metrics[metric].append(value)
        else:
            raise ValueError("path error")
    for metric, value in metrics.items():
        overall_metrics[metric] = np.mean(overall_metrics[metric])
        print(f"{metric}: {overall_metrics[metric]:.4f}")


if __name__ == "__main__":
    main()
