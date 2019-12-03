"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
import cv2
# from processor import process_image
# from keras.utils import to_categorical

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length=40, class_limit=None, video_limit=None, image_shape=(224, 224, 3), skip_rate=10,
                 class_list=None):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        video_limit = (int) number of videos per class to be used
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        # max_frames has no use for us
        # self.max_frames = 300  # max number of frames a video can have for us to use it
        self.image_shape = image_shape
        self.skip_rate = skip_rate
        if class_list is None:
            self.class_list = None
        else:
            self.class_list = []
            [self.class_list.append(x) for x in class_list if x not in self.class_list]  # Remove any duplicate classes
        if video_limit is None:
            self.video_limit = np.Inf
        else:
            self.video_limit = video_limit
        self.class_count = None

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()



    @staticmethod
    def get_data():
        """Load our data from file."""
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        data.sort(key=lambda x: x[1], reverse=False)  # Sort in place by category
        data.sort(key=lambda x: x[0], reverse=True)  # Sort in place by test/train

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        self.class_count = {class_name: 0 for class_name in self.classes}
        for item in self.data:
            if int(item[3]) >= self.seq_length * self.skip_rate \
                    and item[1] in self.classes and self.class_count[item[1]] < self.video_limit:
                data_clean.append(item)
                self.class_count[item[1]] += 1

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        if self.class_list is not None:
            possible_classes = classes
            classes = list()
            # Add desired class to classes list if it is available.
            [classes.append(x) for x in self.class_list if x in possible_classes]

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def video_to_vid_array(self, input_video):
        """

        :param input_video:
        :return: sequence: numpy array shape (sequence length, C, H, W)
        """
        frames = self.get_frames_for_sample(input_video)  # Get the frames for this video.
        frames = self.rescale_list(frames, self.seq_length)  # Now downsample to just the ones we need.
        sequence = np.zeros((self.seq_length, self.image_shape[2], self.image_shape[0], self.image_shape[1]))
        for index, frame in enumerate(frames):
            image_array = self.pic_to_pic_array(frame)
            image_array = np.expand_dims(image_array, 0)  # Add sequence dimension
            sequence[index] = image_array

        return sequence

    def pic_to_pic_array(self, image_location):
        """
        Loads image to array, crops array to simage shape and converts image to C, H, W.
        :param image_location:
        :return:
        """
        image_array = cv2.imread(image_location)  # Convert image to array
        H, W, C = image_array.shape
        if H > self.image_shape[0]:
            low_side = int(np.floor((H - self.image_shape[0]) / 2.0))
            high_side = int(np.ceil((H - self.image_shape[0]) / 2.0))
            image_array = image_array[low_side: -high_side]
        if W > self.image_shape[1]:
            low_side = int(np.floor((W - self.image_shape[1]) / 2.0))
            high_side = int(np.ceil((W - self.image_shape[1]) / 2.0))
            image_array = image_array[:, low_side: -high_side]
        image_array = np.transpose(image_array, (2, 0, 1))  # Convert to C, H, W

        return image_array

    def vid_array_to_video(self, filename, video_array):
        for index in range(video_array.shape[0]):
            self.pic_array_to_pic(filename + "saved_{:04}.jpg".format(index), video_array[index])

    def pic_array_to_pic(self, filename, picture_array):
        picture_array = np.transpose(picture_array, (1, 2, 0))  # Convert to H, W, C
        cv2.imwrite(filename, picture_array)

    # def get_class_one_hot(self, class_str):
    #     """Given a class as a string, return its number in the classes
    #     list. This lets us encode and one-hot it for training."""
    #     # Encode it first.
    #     label_encoded = self.classes.index(class_str)
    #
    #     # Now one-hot it.
    #     label_hot = to_categorical(label_encoded, len(self.classes))
    #
    #     assert len(label_hot) == len(self.classes)
    #
    #     return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    # def build_image_sequence(self, frames):
    #     """Given a set of frames (filenames), build our sequence."""
    #     return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join('data', sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    # @staticmethod
    def rescale_list(self, input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        # skip = len(input_list) // size  # TODO: Make a better skip

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), self.skip_rate)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
