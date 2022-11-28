import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from architectures import *


def compute_weights(labels):
    """
    Compute the weights for each class. The higher the number of samples for a class, the lower the weight.

    Parameters
    ----------
    labels : numpy.ndarray
        The labels of the dataset.

    Returns
    -------
    dict_weights : dict
        The weights for each class.
    """
    labels = np.argmax(labels, axis=-1)

    occurrences = []
    for i in np.unique(labels):
        occurrences.append(np.sum(labels == i))

    weights = []

    for occurrence in occurrences:
        weight = 1 + (1 - occurrence / np.max(occurrences))
        weights.append(weight)

    dict_weights = {}
    for i in np.unique(labels):
        dict_weights.update({i: weights[i]})

    return dict_weights


def get_f1(y_true, y_pred):
    """
        Computes the F1 score, the harmonic mean of precision and recall.
        This is a better metric than accuracy, especially if for an uneven class distribution.
        Source: https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
        Source: https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
        Source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        Parameters
        ----------
        y_true : TensorFlow/Theano tensor
            A tensor of the same shape as `y_pred`
        y_pred : TensorFlow/Theano tensor of float type
            A tensor resulting from a sigmoid
        Returns
        -------
        The F1 score as a single tensor.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def get_next_flag(epoch, epoch_flags, epochs):
    """
        get_next_flag(epoch, epoch_flags, epochs)

        Returns the next flag in the epoch_flags list that is greater than the epoch.
        If no such flag exists, returns the epochs value.

        Parameters
        ----------
        epoch : int
            The current epoch.
        epoch_flags : list
            A list of epochs at which to change the learning rate.
        epochs : int
            The total number of epochs.

        Returns
        -------
        next_flag : int
            The next flag in the epoch_flags list that is greater than the epoch.
            If no such flag exists, returns the epochs value.
    """
    next_flag = epochs
    i = 0
    while i < len(epoch_flags):
        if epoch < epoch_flags[i]:
            next_flag = epoch_flags[i]
            break
        i += 1

    return next_flag


def freeze_unfreeze_feature_extractor(model, freeze, name="efficientnetb3"):
    """
        Freeze or unfreeze the feature extractor of a model.

        Parameters
        ----------
        model : keras.Model
            The model to freeze or unfreeze.
        freeze : bool
            Whether to freeze or unfreeze the feature extractor.
        name : str
            The name of the feature extractor.

        Returns
        -------
        keras.Model
            The model with the feature extractor frozen or unfrozen.
    """
    model.get_layer(name=name).trainable = not freeze
    return model


def update_weights(scores, classes, power):
    """
        This function takes the scores of each class and returns the weights for each class.
        The weights are calculated based on the f1-score of each class.
        The higher the f1-score, the lower the weight.
        The lower the f1-score, the higher the weight.
        The weights are calculated using the following formula:
            weight = 1 + (1 - f1**power / (np.max(f1s)**power + np.finfo(float).eps))
        The power is a hyperparameter that can be tuned.
        The default value is 2.
        The weights are returned as a dictionary.
        The keys are the class labels.
        The values are the weights.
    """
    f1s = []
    for i in range(classes):
        f1s.append(scores[str(i)]['f1-score'])

    weights = []
    for f1 in f1s:
        weight = 1 + (1 - f1 ** power / (np.max(f1s) ** power + np.finfo(float).eps))
        weights.append(weight)

    dict_weights = {}
    for i in range(classes):
        dict_weights.update({i: weights[i]})

    return dict_weights


def get_binary_labels(Y, species=None):
    """
        This function takes in a matrix of labels and returns a binary matrix of labels.
        If a species is specified, then the function returns a binary matrix of labels
        where the specified species is 1 and all other species are 0.
        If no species is specified, then the function returns the original matrix of labels.
    """
    if species is not None:
        bin_labels = np.zeros([Y.shape[0], 1])
        for i in range(Y.shape[0]):
            if Y[i, species] == 1:
                bin_labels[i, 0] = 1
        return bin_labels
    else:
        return Y


def get_callbacks(phase_epochs):
    """
        This function returns a list of callbacks to be used during training.
        The callbacks are:
            - TensorBoard
            - ModelCheckpoint
            - CSVLogger
            - EarlyStopping
            - CustomCallback
        The TensorBoard callback is used to log the training and validation metrics
        in a TensorBoard compatible format.
        The ModelCheckpoint callback is used to save the model with the best validation
        accuracy.
        The CSVLogger callback is used to log the training and validation metrics
        in a CSV file.
        The EarlyStopping callback is used to stop the training when the validation
        loss does not improve for a certain number of epochs.
        The CustomCallback callback is used to print the current learning rate
        at the beginning of each epoch.
        Args:
            phase_epochs: The number of epochs for the current training phase.
        Returns:
            A list of callbacks to be used during training.
    """

    tboard = 'tb_logs'
    os.makedirs(tboard, exist_ok=True)
    tb_call = TensorBoard(log_dir=tboard)

    chkpt_dir = 'float_model'
    os.makedirs(chkpt_dir, exist_ok=True)
    chkpt_call = ModelCheckpoint(
        filepath=os.path.join(chkpt_dir, '{val_accuracy:.4f}-{accuracy:.4f}-f_model.h5'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True)

    logdir = 'train_log.csv'
    csv_logger = CSVLogger(logdir, append=True, separator=';')

    patience = phase_epochs - int(0.6 * phase_epochs)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.05, patience=patience,
                                                      restore_best_weights=False)

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            current_decayed_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            print("current decayed lr: {:0.7f}".format(current_decayed_lr))

    return [tb_call, chkpt_call, csv_logger, early_stopping, CustomCallback()]


def variable_training(model, train_dataset, val_dataset, data_gen, epochs: int, epoch_flags: (int, list), learn_rates,
                      loss_functions, class_weights: dict, adjust_weights, classes: int,
                      frozen_FE: bool):
    """
    This function trains a model with variable learning rate, loss function and class weights.
    It is possible to freeze the feature extractor.
    The training is divided in phases, each one with its own learning rate, loss function and class weights.
    The phases are defined by epoch_flags, which is a list of epochs where the training parameters change.
    The last phase is defined by the last epoch_flag and the number of epochs.
    The learning rate is defined by learn_rates, which is a list of learning rates for each phase.
    The loss function is defined by loss_functions, which is a list of loss functions for each phase.
    The class weights are defined by class_weights, which is a dictionary of weights for each class.
    The feature extractor is frozen if frozen_FE is True, and it is unfrozen if frozen_FE is False.
    If adjust_weights is True, the class weights are updated at the end of each phase.
    The number of classes is defined by classes.
    The model is trained on train_dataset and validated on val_dataset.
    The data generator is data_gen.
    The number of epochs is defined by epochs.
    The model is returned.
    """

    if type(epoch_flags) == int:
        assert epoch_flags < epochs
        step = epoch_flags
        epoch_flags = [step]
        while epoch_flags[-1] < epochs:
            epoch_flags.append(epoch_flags[-1] + step)
        epoch_flags.pop()
    else:
        epoch_flags = sorted(epoch_flags)
        assert epoch_flags[-1] < epochs

    phases = len(epoch_flags) + 1
    print(epoch_flags)

    if type(learn_rates) == list:
        assert len(learn_rates) == phases + 1  # I need the end_learning_rate for last phase
    elif type(learn_rates) == float:
        learn_rates = [learn_rates] * (phases + 1) # I need the end_learning_rate for last phase
    else:
        assert callable(learn_rates)  # check it is a function

    print(learn_rates)

    if not type(loss_functions) == list:
        loss_functions = [loss_functions] * phases
    assert len(loss_functions) == phases
    for loss in loss_functions:
        assert callable(loss)

    print(loss_functions)

    if type(class_weights) == dict:
        assert len(class_weights) == classes
    else:
        class_weights = {}
        for i in range(classes):
            class_weights.update({i: 1})

    print(class_weights)

    if type(frozen_FE) == bool:
        frozen_FE = [frozen_FE] * phases
    elif type(frozen_FE) == list:
        assert len(frozen_FE) == phases
        for fr_FE in frozen_FE:
            assert type(fr_FE) == bool
    else:
        raise AttributeError

    print(frozen_FE)

    epoch = 0
    print(model.summary())

    for phase in range(phases):
        print("Starting phase ", phase)

        end_of_phase_epoch = get_next_flag(epoch, epoch_flags, epochs)

        print("Epochs from ", epoch, " to ", end_of_phase_epoch)

        model = freeze_unfreeze_feature_extractor(model, frozen_FE[phase])
        phase_epochs = end_of_phase_epoch - epoch
        end_learning_rate = learn_rates[phase + 1]
        learning_rate_fn = PolynomialDecay(
            learn_rates[phase],
            decay_steps=89 * phase_epochs, #89 steps per epochs with batch_size = 32
            end_learning_rate=end_learning_rate,
            power=2.3)

        metrics = ['accuracy'] if classes > 2 else ['accuracy', get_f1]
        print("compiling model")
        model.compile(optimizer=Adam(learning_rate=learning_rate_fn),
                      loss=loss_functions[phase],
                      metrics=metrics)

        print("fitting model")
        callbacks = get_callbacks(phase_epochs)
        model.fit(train_dataset,
                  epochs=phase_epochs,
                  validation_data=val_dataset,
                  callbacks=callbacks,
                  class_weight=class_weights,
                  verbose=1)

        X = []
        Y = []
        for i in range(val_dataset.__len__()):
            x, y = val_dataset.__getitem__(i)
            X.append(x)
            Y.append(y)

        X = np.concatenate(X)
        y_true = np.concatenate(Y)
        prediction_data_gen = ImageDataGenerator()
        prediction_data = prediction_data_gen.flow(X, batch_size=32, shuffle=False)

        y_pred = model.predict(prediction_data)

        scores = classification_report(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), digits=4,
                                       output_dict=True)

        print(classification_report(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), digits=4,
                                    output_dict=False))
        if adjust_weights:
            class_weights = update_weights(scores, classes, power=4)
            print("Weights updated to: ")
            print(class_weights)

        epoch = end_of_phase_epoch

    return model


if __name__ == '__main__':

    seed = 1

    data_gen = ImageDataGenerator(rotation_range=180,
                                  height_shift_range=0.1,
                                  width_shift_range=0.1,
                                  zoom_range=0.1,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='reflect',
                                  brightness_range=[0.8, 1.2],
                                  validation_split=0.2)

    dataset_dir = 'dataset'
    IMG_SIZE = 96
    classes = 8
    train_dataset = data_gen.flow_from_directory(directory=dataset_dir,
                                                 target_size=(IMG_SIZE, IMG_SIZE),
                                                 color_mode='rgb',
                                                 classes=None,  # can be set to labels
                                                 class_mode='categorical',
                                                 batch_size=32,
                                                 subset='training',
                                                 seed=seed)

    val_dataset = ImageDataGenerator(validation_split=0.2).flow_from_directory(directory=dataset_dir,
                                                                               target_size=(IMG_SIZE, IMG_SIZE),
                                                                               color_mode='rgb',
                                                                               classes=None,  # can be set to labels
                                                                               class_mode='categorical',
                                                                               batch_size=32,
                                                                               subset='validation',
                                                                               seed=seed)

    model = get_EfficientNetB3(classes=classes, weights="imagenet", l1=0.00001, l2=0.00001)

    learn_rates = [0.001, 0.0006, 3.6784e-04, 2.1350e-04, 1.2595e-04, 8.0690e-05, 6.0140e-05, 5.2440e-05,
                   5.0330e-05, 5.0011e-05]

    loss = categorical_crossentropy

    cl_w = {}
    for i in range(classes):
        cl_w.update({i: 1})

    frozen_FE = [True] + [False] * 8
    variable_training(model, train_dataset, val_dataset, data_gen, epochs=900, epoch_flags=100,
                      learn_rates=learn_rates, loss_functions=loss, class_weights=cl_w, adjust_weights=True,
                      classes=classes, frozen_FE=frozen_FE)
