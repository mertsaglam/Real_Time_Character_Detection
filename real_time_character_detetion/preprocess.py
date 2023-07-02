import cv2
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Load the data
dataset = pd.read_csv('A_Z Handwritten Data.csv').astype('float32')
dataset.rename(columns={'0': "label"}, inplace=True)
#split the label and data, load digit dataset
letter_x = dataset.drop("label", axis=1)
letter_y = dataset["label"]
(digit_train_x, digit_train_y), (digit_test_x, digit_test_y) = mnist.load_data()
letter_x = letter_x.values
print("letter_x shape: ", letter_x.shape, "letter_y shape: ", letter_y.shape)
print("digit_train_x shape: ", digit_train_x.shape, "digit_train_y shape: ", digit_train_y.shape)
print("digit_test_x shape: ", digit_test_x.shape, "digit_test_y shape: ", digit_test_y.shape)
#concetanate the test and train
digit_data = np.concatenate((digit_train_x, digit_test_x))
digit_target = np.concatenate((digit_train_y, digit_test_y))

print("digit_data shape: ", digit_data.shape, "digit_targets shape: ", digit_target.shape)

#update digit target
digit_target = digit_target + 26

data = []
for flatten in letter_x:
    image = np.reshape(flatten, (28, 28,1))
    data.append(image)
letter_data = np.array(data, dtype="float32")
letter_target = letter_y

digit_data = np.reshape(digit_data, (digit_data.shape[0], 28, 28, 1))

print("letter_data shape: ", letter_data.shape, "letter_target shape: ", letter_target.shape)
print("digit_data shape: ", digit_data.shape, "digit_target shape: ", digit_target.shape)




plt.rcParams["font.family"] = "Arial"

indices = np.arange(letter_data.shape[0])
np.random.shuffle(indices)

shuffled_data = letter_data[indices]
rows, cols = 10, 10

plt.figure(figsize=(20, 20))

for i in range(rows * cols):
    plt.subplot(cols, rows, i+1)
    plt.imshow(shuffled_data[i].reshape(28, 28), interpolation="nearest", cmap="gray")

plt.show()

shuffled_data = shuffle(digit_data)


rows, cols = 10, 10

plt.figure(figsize=(20, 20))

for i in range(rows * cols):
  plt.subplot(cols, rows, i+1)
  plt.imshow(shuffled_data[i].reshape(28, 28), interpolation="nearest", cmap="gray")

plt.show()

#concatenate digit and letter data

data = np.concatenate((digit_data, letter_data))
target = np.concatenate((digit_target, letter_target))

print(data.shape, target.shape)


label_mapping = {i: chr(65 + i) if i < 26 else chr(48 + i - 26) if i < 36 else ' ' for i in range(37)}
shuffled_data, shuffled_targets = shuffle(data,target)
rows, cols = 10, 10

plt.figure(figsize=(20, 20))

for i in range(rows * cols):
  plt.subplot(cols, rows, i+1)
  plt.imshow(shuffled_data[i].reshape(28, 28), interpolation="nearest", cmap="gray")
  plt.title('Label: {}'.format(label_mapping[int(shuffled_targets[i])]))


plt.show()

#split the data into train and test
train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.2, random_state=42)

print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)

#normalize the data
train_data = train_data / 255.0
test_data = test_data / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)


train_label_counts = [0 for i in range(36)]
test_label_counts = [0 for i in range(36)]

for i in range(train_data.shape[0]):
  train_label_counts[np.argmax(train_labels[i])] += 1

for i in range(test_data.shape[0]):
  test_label_counts[np.argmax(test_labels[i])] += 1

#plot the data

frequency = [
    train_label_counts, test_label_counts
] 

fig =  plt.figure(figsize=(8, 6))
ax = fig.add_axes([0, 0, 1, 1])
x = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.xticks(range(len(frequency[0])), x)
plt.title("train vs. test data distribution")
plt.xlabel("character")
plt.ylabel("frequency")

ax.bar(np.arange(len(frequency[0])), frequency[0], color="b", width=0.35)
ax.bar(np.arange(len(frequency[1])) + 0.35, frequency[1], color="r", width=0.35)
ax.legend(labels=["train", "test"])

#save the data
np.save("train_data", train_data)
np.save("train_labels", train_labels)
np.save("test_data", test_data)
np.save("test_labels", test_labels)
