from keras.layers import Conv2D, Lambda, Cropping2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from random import shuffle
import csv
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

steering_correction = 0.1
learning_rate = 0.001


def build_model(input_shape, output):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # Crops off the hood and sky
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model


# Generator that loads the data when needed instead of all at once
def generator(data, batch_size=32):
    num_samples = len(data)
    while 1:
        shuffle(data)

        for i in range(0, num_samples, batch_size):
            batch = data[i:i+batch_size]

            images = []
            angles = []
            for sample in batch:
                img_center, img_left, img_right, ang_center, ang_left,\
                    ang_right = get_data(sample)

                images.append(img_center)
                images.append(img_left)
                images.append(img_right)

                angles.append(ang_center)
                angles.append(ang_left)
                angles.append(ang_right)

                img_center_aug, img_left_aug, img_right_aug, ang_center_aug, ang_left_aug, \
                    ang_right_aug = get_data(sample, augment=True)

                images.append(img_center_aug)
                images.append(img_left_aug)
                images.append(img_right_aug)

                angles.append(ang_center_aug)
                angles.append(ang_left_aug)
                angles.append(ang_right_aug)

            x = np.array(images)
            y = np.array(angles)

            yield x, y


def get_data(data, augment=False):
    img_center = cv2.cvtColor(cv2.imread(data[0]), cv2.COLOR_BGR2RGB)
    img_left = cv2.cvtColor(cv2.imread(data[1]), cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(cv2.imread(data[2]), cv2.COLOR_BGR2RGB)

    ang_center = data[3]
    ang_left = data[4]
    ang_right = data[5]

    if not augment:
        return img_center, img_left, img_right, ang_center, ang_left, ang_right
    else:
        return cv2.flip(img_center, 1), cv2.flip(img_left, 1), cv2.flip(img_right, 1),\
                ang_center * - 1.0, ang_left * - 1.0, ang_left * - 1.0


def prep_data(line, subfolder):
    # Prep our data
    da = []

    # Add the filenames
    for i in range(3):
        fname = "data/" + subfolder + line[i].split("data")[-1]
        da.append(fname)

    # Add the steering angles
    da.append(float(line[3]))
    da.append(float(line[3]) + steering_correction)
    da.append(float(line[3]) + steering_correction)

    return da


def train():
    # Get all the file names and measurements
    dat = []
    for d in os.listdir("data/"):
        if d != "driving_log.csv" and d != "IMG":
            with open("data/" + d + "/driving_log.csv") as csvFile:
                reader = csv.reader(csvFile)
                for l in reader:
                    dat.append(prep_data(l, d))

    test_image = cv2.imread(dat[0][0])
    # Create and train our model
    model = build_model(input_shape=test_image.shape, output=1)
    #model = load_model("model.h5")

    # Create train and test samples
    train_data, val_data = train_test_split(dat, test_size=0.2)

    # Create generators
    train_gen = generator(train_data, batch_size=24)
    val_gen = generator(val_data, batch_size=24)

    # Train the model
    model.fit_generator(train_gen, len(train_data), validation_data=val_gen,
                        validation_steps=len(val_data), epochs=6, verbose=1)
    model.save("model.h5")
    print("model saved")

"""
def render_image(image, name=""):
    cv2.imshow(name, image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


# Get all the file names and measurements
dat = []
for d in os.listdir("data/"):
    if d != "driving_log.csv" and d != "IMG":
        with open("data/" + d + "/driving_log.csv") as csvFile:
            reader = csv.reader(csvFile)
            for l in reader:
                dat.append(prep_data(l, d))

# Load a test image to get our input shape
print(dat[0])
test_image = cv2.imread(dat[0][0])
test_image_left = cv2.imread(dat[0][1])
test_image_right = cv2.imread(dat[0][2])
while True:
    render_image(test_image, "center")
    render_image(test_image_left, "left")
    render_image(test_image_right, "right")"""

train()