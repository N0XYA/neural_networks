import tensorflow_datasets as tfds


ds = tfds.load("cats_vs_dogs", split="train")
for ex in ds.take(4):
    print(ex)