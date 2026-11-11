"""
Preprocess multihop_qa dataset for ROSA training.
"""
import argparse
import pandas as pd
import datasets
import numpy as np


def preprocess_multihop_qa(input_path: str, output_path: str):
    """
    Preprocess multihop_qa parquet file to add required fields.

    Extracts question from prompt and answer from reward_model.ground_truth,
    then adds them to extra_info.
    """
    # Load the parquet file
    df = pd.read_parquet(input_path)

    # Determine split name from path
    split_name = "train" if "train" in input_path else "test"

    new_data = {
        "data_source": [],
        "prompt": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
    }

    for idx, row in df.iterrows():
        # Extract question from prompt content
        prompt = row["prompt"]
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        if isinstance(prompt, list) and len(prompt) > 0:
            question = prompt[0].get("content", "") if isinstance(prompt[0], dict) else str(prompt[0])
        else:
            question = str(prompt)

        # Extract answer from reward_model.ground_truth.target
        reward_model = row["reward_model"]
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", {})
            if isinstance(ground_truth, dict):
                answer = ground_truth.get("target", "")
                # If target is a list, join it
                if isinstance(answer, np.ndarray):
                    answer = answer.tolist()
                if isinstance(answer, list):
                    answer = " ".join(str(a) for a in answer)
            else:
                answer = str(ground_truth)
        else:
            answer = ""

        new_data["data_source"].append(row.get("data_source", "multihop_qa"))
        new_data["prompt"].append(prompt)
        new_data["ability"].append(row.get("ability", "reasoning"))
        new_data["reward_model"].append(reward_model)
        new_data["extra_info"].append({
            "split": split_name,
            "index": idx,
            "answer": answer,
            "question": question,
        })

    # Create dataset
    dataset = datasets.Dataset.from_dict(new_data)

    # Save to parquet
    dataset.to_parquet(output_path)
    print(f"Saved preprocessed data to {output_path}")
    print(f"Total samples: {len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output parquet file")
    args = parser.parse_args()

    preprocess_multihop_qa(args.input_path, args.output_path)
