import tensorflow_datasets as tfds


ds = tfds.load("cats_vs_dogs", split="train")
ds = ds.take(1)
for example in ds:
    print(list(example.keys()))
    print(type(example))
    image = example["image"]
    label = example["label"]
    print(image.shape, label)
    print(type(image))
    