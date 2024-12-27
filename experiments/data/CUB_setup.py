import pandas as pd

# expects dataset to be unzipped into data directory
PATH = "CUB_200_2011/attributes/image_attribute_labels.txt"
TRAIN_TEST_PATH = "CUB_200_2011/train_test_split.txt"

# read csv
df = pd.read_csv(
    PATH,
    sep=r"\s+",
    names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
    on_bad_lines="warn",
)

# reshape
data = df.pivot_table(index="image_id", columns="attribute_id", values="is_present")

data.to_csv("CUB_200_2011_concepts.csv")


# train/test split
train_indices = pd.read_csv(
    TRAIN_TEST_PATH, sep=r"\s+", names=["image_id", "is_train"], index_col="image_id"
)

data.loc[train_indices["is_train"] == 1].to_csv("CUB_200_2011_concepts_train.csv")
data.loc[train_indices["is_train"] == 0].to_csv("CUB_200_2011_concepts_test.csv")
