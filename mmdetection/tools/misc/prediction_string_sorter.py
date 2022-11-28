import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Sort PredictionString in order")
    parser.add_argument('in_file', help='path of the submission file to fix')
    parser.add_argument('-s', '--save', help='path to save the amended file')
    parser.add_argument('-O', '--overwrite', action='store_true', help='overwrite old file with new file')
    args = parser.parse_args()
    return args


def sort_string(in_file, out_file):
    ord = lambda pred: (pred[0], -pred[1])

    sub_in = pd.read_csv(in_file)
    image_ids = sub_in['image_id'].tolist()

    sorted_preds_lists = []
    for id in range(len(image_ids)):
        sorted_preds_list = []
        sorted_preds_str = ''

        preds = str(sub_in['PredictionString'][id]).split()
        preds = np.array(preds, dtype=float)
        preds = np.reshape(preds, (-1, 6))

        for pred in preds:
            sorted_preds_list.append(
                [int(pred[0]), pred[1], pred[2], pred[3], pred[4], pred[5]]
            )

        sorted_preds_list.sort(key=ord)

        for pred_list in sorted_preds_list:
            for pred in pred_list:
                sorted_preds_str += str(pred) + ' '
        
        sorted_preds_lists.append(sorted_preds_str)

    submission_to = pd.DataFrame()
    submission_to['PredictionString'] = sorted_preds_lists
    submission_to['image_id'] = image_ids

    submission_to.to_csv(out_file, index=False)


def main():
    args = parse_args()

    if not args.overwrite:
        assert args.save is not None, "Overwrite option is off. Path to save new file is required."
        assert args.in_file != args.save, "Overwrite option is off. Input and output file path sould be different."
        input_path = args.in_file
        output_path = args.save
    else:
        input_path = args.in_file
        output_path = args.in_file
    
    sort_string(input_path, output_path)


if __name__ == '__main__':
    main()
