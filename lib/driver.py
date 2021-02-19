from lib.classify import Classify
from lib.timit import Timit
from lib.arch import networks
from lib.buildmodels import build_model
from keras.utils import np_utils
from lib.augmentator import augmentate

from sklearn.model_selection import KFold
import numpy as np
import keras
import matplotlib.pyplot as plt


def driver():
    """
    driver() will fit a deep neural network to predict what digits are being spoken in audio files.
    """

    # the following parameters are used to fine tune the deep neural network
    adv_ms = 12
    len_ms = 25
    avg_frames_per_speech_duration = 42
    width = 100
    penalty = 0.001
    drop_rate = 0.2
    batch_size = 32
    epochs = 15
    n_folds = 5
    optimizer = "Adam"
    network_name = "DenseX4_L2"
    augment_data = True
    output_file = "output/statistics.txt"

    # set dataset base directory
    dataset_basedir = "../tidigits-isolated-digits-wav/wav/"

    # if data augmentation set to True, generate the three different types of data augmentation
    if augment_data:
        print("Augmenting Data ... ")
        augmentate(dataset_basedir, "add_noise")
        augmentate(dataset_basedir, "stretch_time")
        augmentate(dataset_basedir, "shift_pitch")

    # prepare the dataset, grab the training and validation dataset, the testing dataset and its labels
    dataset = Timit(dataset_basedir)
    training_and_validation_data = np.array(dataset.get_filenames("train", augment_data))
    np.random.shuffle(training_and_validation_data)
    testing_data = dataset.get_filenames("test")
    testing_labels = np_utils.to_categorical(dataset.filename_to_class(testing_data)).astype(int)

    # prepare classifier, prepare for model training results, then proceed to model training and testing
    classifier = Classify(adv_ms, len_ms, avg_frames_per_speech_duration)
    models = []
    errors = []
    losses = []

    # Create a plan for k-fold testing with shuffling of examples
    kfold = KFold(n_folds, shuffle=True)

    # run kfold cross validation
    features_idx = range(len(training_and_validation_data))
    for (training_idx, validation_idx) in kfold.split(features_idx):
        # split into training and validation datasets
        training_data = training_and_validation_data[training_idx]
        validation_data = training_and_validation_data[validation_idx]
        # use one hot vector encoding for categorical prediction
        training_labels = np_utils.to_categorical(dataset.filename_to_class(training_data)).astype(int)
        validation_labels = np_utils.to_categorical(dataset.filename_to_class(validation_data)).astype(int)
        # grab the desired model design, build the model, compile it, and print a summary
        network_gen = networks[network_name]
        # 10500 comes from 42*250, the size of the flatted dft stream for an audio file
        # I hard coded this here to make it simpler, but ideally it shouldn't be
        network_list = network_gen(width, 10500, len(dataset.get_class_labels()), penalty)
        network = build_model(network_list)
        network.compile(optimizer=optimizer,
                        loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_accuracy])
        network.summary()
        # train the model, test it, and store the model, error rate, and loss, all using the classifier
        network, loss = classifier.train(network, training_data, validation_data, training_labels, validation_labels,
                                   batch_size, epochs)
        error_rate = classifier.test(network, network_name, validation_data, validation_labels, len(validation_labels),
                                     dataset.get_class_labels())
        models.append(network)
        errors.append(error_rate)
        losses.append(loss)

    # determine which model performed the best
    min_error_index = errors.index(min(errors))

    # test best model on test data, and also create confusion matrix for it
    test_error_rate = classifier.test(models[min_error_index], network_name, testing_data, testing_labels,
                                      len(testing_labels), dataset.get_class_labels(), plot_conf=True)

    # plot the loss per epoch of best model and save as png
    plt.figure()
    plt.plot(losses[min_error_index])
    plt.title("Loss for Fold {}".format(min_error_index + 1))
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("output/loss.png")

    # write results to output file.
    # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    # I referenced the above code to print the model summary to a string in order to write to the output file
    output = open(output_file, 'w')
    model_list = []
    models[min_error_index].summary(print_fn=lambda layer: model_list.append(layer))
    short_model_summary = "\n".join(model_list)
    output.write(short_model_summary + '\n')

    # output error rate of each fold from cross validation
    for index, error in enumerate(errors):
        output.write("Fold {} has an error rate of {}\n".format(index + 1, error))

    # output error mean and standard deviation
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    output.write("Error rate mean: {}\n".format(error_mean))
    output.write("Error rate standard deviation: {}\n".format(error_std))

    # output error rate on the test data
    output.write("Error rate for fold {} on TIDIGIT test data: {}\n\n".format(min_error_index + 1, test_error_rate))
    output.close()
    print("Done")


if __name__ == "__main__":
    driver()
