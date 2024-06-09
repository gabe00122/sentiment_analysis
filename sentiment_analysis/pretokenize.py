import numpy as np
import json
from sentencepiece import SentencePieceProcessor

# import pyarrow as pa
# import pyarrow.parquet as pq


def main():
    max_length = 128

    processor = SentencePieceProcessor(model_file="l.model")
    data_path = (
        "/Users/gabrielkeith/Downloads/yelp_dataset/yelp_academic_dataset_review.json"
    )

    stars_data = []
    length_data = []
    tokens_data = []

    with open(data_path, "r") as data_file:
        for line in data_file:
            json_data = json.loads(line)
            text = json_data["text"].lower()
            stars = int(json_data["stars"])
            tokens = processor.encode(text, out_type=int)
            length = len(tokens)

            if length <= max_length:
                stars_data.append(stars)
                length_data.append(length)
                tokens_data.append(tokens + [-1] * (max_length - length))

    # schema = pa.schema(
    #     [
    #         ("stars", pa.int16()),
    #         ("length", pa.int16()),
    #         ("tokens", pa.list_(pa.int16(), max_length)),
    #     ]
    # )
    # table = pa.table(
    #     [stars_data, length_data, tokens_data],
    #     schema=schema,
    # )
    # pq.write_table(table, "data/training_data.parquet")

    # print(table)
    tokens = np.array(tokens_data, np.int16)
    stars = np.array(stars_data, np.int16)
    lengths = np.array(length_data, np.int16)
    np.savez_compressed(
        "data/training_data.npz", tokens=tokens, stars=stars, lengths=lengths
    )


if __name__ == "__main__":
    main()
