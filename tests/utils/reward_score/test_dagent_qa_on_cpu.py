from verl.utils.reward_score import default_compute_score


def test_reason_search_dataset_uses_qa_reward() -> None:
    ground_truth = {"target": ["Paris"]}

    assert default_compute_score(
        "dagent_reason_search_wiki18", "<answer>Paris</answer>", ground_truth
    ) == 1.0
    assert default_compute_score(
        "dagent_reason_search_wiki18", "<answer>London</answer>", ground_truth
    ) == 0.0


if __name__ == "__main__":
    test_reason_search_dataset_uses_qa_reward()
