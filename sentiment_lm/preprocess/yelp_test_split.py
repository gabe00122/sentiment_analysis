import random
from pathlib import Path
import json

data_path = "./data/yelp_academic_dataset_review.json"
training_path = "./data/training.json"
test_path = "./data/test.json"
validation_path = "./data/validation.json"


def train_test_split(
    test_ratio: float = 0.2,
    validation_ratio: float = 0.05,
    seed: int = 1234,
):
    random.seed(seed)
    training_count = 0
    test_count = 0
    validation_count = 0

    def print_stats():
        print(f"training_count = {training_count}")
        print(f"test_count = {test_count}")
        print(f"validation_count = {validation_count}")

    with (
        open(data_path, "r") as data_file,
        open(training_path, "w") as training,
        open(test_path, "w") as test,
        open(validation_path, "w") as validation,
    ):
        for i, line in enumerate(data_file):
            data = json.loads(line)
            out_data = json.dumps({"text": data["text"], "label": data["stars"]}) + "\n"
            sample = random.uniform(0, 1)

            if sample <= validation_ratio:
                validation.write(out_data)
                validation_count += 1
            elif sample <= test_ratio + validation_ratio:
                test.write(out_data)
                test_count += 1
            else:
                training.write(out_data)
                training_count += 1

            if i % 10_000 == 9_999:
                print_stats()

    print("done!")


def main():
    train_test_split()


if __name__ == "__main__":
    main()
