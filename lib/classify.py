from lib.generator import Generator
from lib.confusion import plot_confusion
from keras.callbacks import TensorBoard
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os


class Classify:
    """
    A classifier class to train and test models
    """

    def __init__(self, adv_ms, len_ms, avg_frames_per_speech_duration):
        """
        Store the audio frame length and advance, and number of frames to be used in each audio file.  Also,
        since Classify produces many results desired for later analysis, set up output directory for later storage.
        """

        self.adv_ms = adv_ms
        self.len_ms = len_ms
        self.avg_frames_per_speech_duration = avg_frames_per_speech_duration
        if not os.path.isdir("output/"):
            os.mkdir("output/")

    def train(self, model, training_data, validation_data, training_labels, validation_labels, batch_size, epochs):
        """
        train the model, given and training and validation batch generator, along with their labels, the batch size,
        and the number of epochs to run
        """

        # create the training batch generator
        training_generator = Generator(training_data, training_labels, self.adv_ms, self.len_ms,
                                       self.avg_frames_per_speech_duration, batch_size)
        # create the validation batch generator
        validation_generator = Generator(validation_data, validation_labels, self.adv_ms, self.len_ms,
                                         self.avg_frames_per_speech_duration, batch_size)
        # set up so that tensorboard can be used while model is training
        log_dir = "logs/{}".format(time.strftime('%d%b-%H%M'))
        tensor_board = TensorBoard(log_dir=log_dir, write_graph=True, write_grads=True)
        # train the model
        history = model.fit(training_generator, validation_data=validation_generator, epochs=epochs,
                       callbacks=[tensor_board])
        loss = history.history["loss"]
        # return the model and its training loss
        return model, loss

    def test(self, model, model_name, testing_data, testing_labels, batch_size, class_labels, plot_conf=False):
        """
        test the model, given testing data, and the class labels
        """

        # build a testing batch generator, but don't shuffle data, and have it just return all the testing data
        # in one batch
        testing_generator = Generator(testing_data, testing_labels, self.adv_ms, self.len_ms,
                                      self.avg_frames_per_speech_duration, batch_size, shuffle=False)
        # make predictions using trained model
        predictions = model.predict(testing_generator)
        # determine which class label had the highest probability, and plot confusion matrix with actual test labels
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(testing_labels, axis=1)
        # determine the error rate by taking the number of wrong predictions over the total number of predictions
        error_rate = np.count_nonzero(y_pred != y_true) / len(testing_labels)
        error_rate = math.floor(error_rate * 10000) / 100.0
        # if a confusion matrix is desired, plot it, and save it as a png
        if plot_conf:
            plot_confusion(y_pred, y_true, class_labels)
            # show confusion matrix with error rate for testing data once model is trained
            plt.title("{} with Error Rate: {}%".format(model_name, error_rate))
            plt.savefig("output/confusion_matrix.png")
        # return the error rate
        return error_rate


