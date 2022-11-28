import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class model:
    def __init__(self, path):
        self.model1 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6420-0.2257-f_model.h5'))
        self.model2 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6629-0.2384-f_model.h5'))
        self.model3 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/BestPlus1-f_model.h5'))
        self.model4 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/0.6691-0.2051-f_model.h5'))

    def predict(self, X):

        dg = ImageDataGenerator(rotation_range=20,
                                height_shift_range=0.1,
                                width_shift_range=0.1,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='reflect',
                                brightness_range=[0.8, 1.2])

        dg = dg.flow(X, shuffle=False)

        out1 = self.model1.predict(dg)
        out2 = self.model2.predict(dg)
        out3 = self.model3.predict(dg)
        out4 = self.model4.predict(dg)

        for i in range(17):
            out1 = out1 + self.model1.predict(dg)
            out2 = out2 + self.model2.predict(dg)
            out3 = out3 + self.model3.predict(dg)
            out4 = out4 + self.model4.predict(dg)

        out1 = out1 / 23
        out2 = out2 / 23
        out3 = out3 / 23
        out4 = out4 / 23

        out = out1 + out2 + out3 + out4
        out = tf.argmax(out, axis=-1)

        return out
