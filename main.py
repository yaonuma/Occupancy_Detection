from __future__ import print_function

import keras
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling1D
from keras.models import Sequential
from keras.utils import np_utils
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

sns.set()  # change theme
plt.style.use('ggplot')
print('keras version ', keras.__version__)


def show_confusion_matrix(validations, predictions, categories):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=categories,
                yticklabels=categories,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def show_basic_dataframe_info(dataframe, preview_rows=20):
    """
    This function shows basic information for the given dataframe
    Args:
        dataframe: A Pandas DataFrame expected to contain data
        preview_rows: An integer value of how many rows to preview
    Returns:
        Nothing
    """

    # Shape and how many rows and columns
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print("First 20 rows of the dataframe:\n")
    # Show first 20 rows
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")
    # Describe dataset like mean, min, max, etc.
    # print(dataframe.describe())


def read_data(file_path):
    """
    This function reads the sensor data from a file
    Args:
        file_path: URL pointing to the CSV file
    Returns:
        A pandas dataframe
    """

    column_names = ['item',
                    'datetime',
                    'temperature',
                    'humidity',
                    'light',
                    'C02',
                    'humidity_ratio',
                    'occupancy']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)

    df.dropna(axis=0, how='any', inplace=True)

    df = df.drop(columns=['item'])

    def to_seconds(x):
        total = 0
        day = x.split('-')
        total += int(day[2].split(' ')[0]) * 86400

        hour = x.split(' ')[1].split(":")
        total += int(hour[0]) * 3600 + int(hour[1]) * 60 + int(hour[2])

        return total

    df['timestamp'] = df.datetime.apply(to_seconds)
    df = df.drop(columns=['datetime'])
    df = df.astype(float)

    return df


def convert_to_float(x):
    try:
        return np.float(x)
    except ValueError:
        return np.nan


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_occupancy(occupancy, data):
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(15, 10))
    plot_axis(ax0, data['timestamp'], data['temperature'], 'temperature')
    plot_axis(ax1, data['timestamp'], data['humidity'], 'humidity')
    plot_axis(ax2, data['timestamp'], data['light'], 'light')
    plot_axis(ax3, data['timestamp'], data['C02'], 'C02')
    plot_axis(ax4, data['timestamp'], data['humidity_ratio'], 'humidity_ratio')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(str(occupancy)+' - Category Sensor Time Series Data', fontsize=16)
    plt.subplots_adjust(top=0.90)
    # print(occupancy,'asdf')
    # fig.suptitle(str(occupancy)+'Category Sensor Time Series Data', fontsize=16)
    plt.show()


def create_segments_and_labels(df, time_steps, step, label_name):
    """
    This function receives a dataframe and returns the reshaped segments
    of sensor data as well as the corresponding labels
    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segments
        labels:
    """


    n_features = 5
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['temperature'].values[i: i + time_steps]
        ys = df['humidity'].values[i: i + time_steps]
        zs = df['light'].values[i: i + time_steps]
        cs = df['C02'].values[i: i + time_steps]
        hrs = df['humidity_ratio'].values[i: i + time_steps]

        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs, cs, hrs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, n_features)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def eda(df):
    df['occupancy'].value_counts().plot(kind='bar',
                                        title='Training Examples by Occupancy Type')
    plt.show()

    for occupancy in np.unique(df["occupancy"]):
        subset = df[df["occupancy"] == occupancy][0:-1]
        plot_occupancy(occupancy, subset)

    return


def get_train_test(df, time_periods, step_distance, label):

    # Transform the labels from String to Integer via LabelEncoder
    le = preprocessing.LabelEncoder()

    # Add a new column to the existing DataFrame with the encoded values
    df[label] = le.fit_transform(df["occupancy"].values.ravel())

    # Normalize features for dataset
    df['temperature'] = feature_normalize(df['temperature'])
    df['humidity'] = feature_normalize(df['humidity'])
    df['light'] = feature_normalize(df['light'])
    df['C02'] = feature_normalize(df['C02'])
    df['humidity_ratio'] = feature_normalize(df['humidity_ratio'])

    # create data objects from the sensor signals
    x, y = create_segments_and_labels(df, time_periods, step_distance, label)

    # Inspect x data
    # print('x shape: ', x.shape)
    # print(x.shape[0], 'training samples')

    # Inspect y data
    # print('y shape: ', y.shape)

    # Set input & output dimensions
    num_time_periods, num_sensors = x.shape[1], x.shape[2]
    num_classes = le.classes_.size
    # print(list(le.classes_))

    # Convert type for Keras otherwise Keras cannot process the data
    x = x.astype("float32")
    y = y.astype("float32")

    # One-hot encoding of y labels (only execute once!)
    y = np_utils.to_categorical(y, num_classes)
    # print('New y shape: ', y.shape)

    return x, y, num_time_periods, num_sensors, num_classes


def build_model(x_train, y_train, num_time_periods, num_sensors, num_classes):

    # 1D CNN neural network
    input_shape = (num_time_periods * num_sensors)
    model_m = Sequential()
    model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
    model_m.add(Conv1D(128, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
    model_m.add(Conv1D(128, 10, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(128, 10, activation='relu'))
    model_m.add(Conv1D(128, 10, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(num_classes, activation='softmax'))

    # used to implement early stopping
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
    ]

    # compile the model
    model_m.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    # Hyper-parameters
    BATCH_SIZE = 100
    EPOCHS = 10

    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model_m.fit(x_train,
                          y_train,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          callbacks=callbacks_list,
                          validation_split=0.2,
                          verbose=1)

    # summarize history for accuracy and loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()

    return model_m


if __name__ == "__main__":

    # read in the data
    df_train = read_data('data/occupancy_data/datatraining.txt')
    df_test = read_data('data/occupancy_data/datatest2.txt')
    print(df_train.shape)

    # explore the training data a bit
    eda(df_train)

    # data preprocessing and training/test generation from signal
    TIME_PERIODS = 80
    STEP_DISTANCE = 80
    CATEGORIES = ["1", "0"]
    LABEL = "occupancyEncoded"
    x_train, y_train, num_time_periods, num_sensors, num_classes = \
        get_train_test(df_train, TIME_PERIODS, STEP_DISTANCE, LABEL)

    # build the model
    model_m = build_model(x_train, y_train, num_time_periods, num_sensors, num_classes)

    # prepare test data
    x_test, y_test, num_time_periods, num_sensors, num_classes = \
        get_train_test(df_test, TIME_PERIODS, STEP_DISTANCE, LABEL)

    # evaluate test data on trained model
    score = model_m.evaluate(x_test, y_test, verbose=1)

    print("\nAccuracy on test data: %0.2f" % score[1])
    print("\nLoss on test data: %0.2f" % score[0])

    # get predictions
    y_pred_test = model_m.predict(x_test)

    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    show_confusion_matrix(max_y_test, max_y_pred_test, CATEGORIES)

    print("\n--- Classification report for test data ---\n")

    print(classification_report(max_y_test, max_y_pred_test))
