import random
from pydantic import BaseModel
from pathlib import Path
import json


class TestSplitConfig(BaseModel):
    test_ratio: float
    validation_ratio: float
    seed: int


def train_test_split(data_path: Path, config: TestSplitConfig):
    random.seed(config.seed)
    training_count = 0
    test_count = 0
    validation_count = 0

    def print_stats():
        print(f"training_count = {training_count}")
        print(f"test_count = {test_count}")
        print(f"validation_count = {validation_count}")

    with (
        open(data_path, "r") as data_file,
        open("./data/training.json", "w") as training,
        open("./data/test.json", "w") as test,
        open("./data/validation.json", "w") as validation,
    ):
        for i, line in enumerate(data_file):
            data = json.loads(line)
            out_data = json.dumps({"text": data["text"], "label": data["stars"]}) + "\n"
            sample = random.uniform(0, 1)

            if sample <= config.validation_ratio:
                validation.write(out_data)
                validation_count += 1
            elif sample <= config.test_ratio + config.validation_ratio:
                test.write(out_data)
                test_count += 1
            else:
                training.write(out_data)
                training_count += 1

            if i % 10_000 == 9_999:
                print_stats()

    print("done!")


def main():
    config_text = Path("./experiments_settings/test_split.json").read_text()
    config = TestSplitConfig.model_validate_json(config_text)
    train_test_split(Path("./data/raw/yelp_academic_dataset_review.json"), config)


if __name__ == "__main__":
    main()
