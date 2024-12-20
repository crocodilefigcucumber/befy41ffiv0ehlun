import pandas as pd

# expects dataset to be unzipped into data directory
PATH = "CUB_200_2011/attributes/image_attribute_labels.txt"

# read csv
df = pd.read_csv(
    PATH,
    sep=r"\s+",
    names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
    error_bad_lines=False,
)

# reshape
data = df.pivot_table(index="image_id", columns="attribute_id", values="is_present")

data.to_csv("CUB_200_2011_concepts.csv")
