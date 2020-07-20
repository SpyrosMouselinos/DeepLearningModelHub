import pymongo
import pickle
from sklearn import datasets
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import Constraint
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

CLIENT = 'mongodb://localhost:27017'
DB = 'ModelHubDB'
DB2CONNECT = 'dlmodels'

iris = datasets.load_iris()
X = iris.data
Y = iris.target
Y = Y.reshape(-1, 1)

es = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)


def unpack(tf_model, training_config, weights):
    restored_model = deserialize(tf_model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


@tf.keras.utils.register_keras_serializable(package='MyPackage', name='ZeroDiagonalConstraint')
class ZeroDiagonalConstraint(Constraint):
    """
    Custom Implementation of the Zero diagonal Constraint
    """

    def __init__(self):
        return

    def call(self, w):
        """
        Return the 0 diag weight matrix
        :param w: The weight matrix
        :return: The constraint matrix
        """
        w = w - tf.linalg.diag(w)
        return w


@tf.keras.utils.register_keras_serializable(package='MyPackage', name='DeterminantReg')
class DetReg(tf.keras.regularizers.Regularizer):
    """
        Regularizes the Determinant of a Weight Matrix to be less than <threshold>
        so it can maintain properties of iterable methods for stability.
    """

    def __init__(self, thres=0.):
        self.thres = thres

    def __call__(self, x):
        return tf.nn.relu(tf.linalg.det(x) - self.thres)

    def get_config(self):
        return {'thres': float(self.thres)}


# Make all Tensorflow Models Picklable
make_keras_picklable()

model = Sequential()
model.add(Dense(units=10, activation='relu', kernel_constraint=ZeroDiagonalConstraint()))
model.add(Dense(units=10, activation='relu', kernel_regularizer=DetReg()))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, Y, batch_size=1, epochs=10, validation_split=0.5,
          callbacks=[es])


def save_model_to_db(model, client, db, dbconn, model_name):
    ### Pickle The Model
    pickled_model = pickle.dumps(model)

    ### Open Client
    myclient = pymongo.MongoClient(client)

    ### Open Database
    mydb = myclient[db]

    ### Establish Connection
    mycon = mydb[dbconn]

    ### Get Info
    info = mycon.insert_one({'ModelName': model_name, 'ModelContent': pickled_model, 'ModelType': 'DNN'})
    return info


def load_saved_model_from_db(model_name, client, db, dbconn):
    json_data = {}
    myclient = pymongo.MongoClient(client)
    mydb = myclient[db]
    mycon = mydb[dbconn]
    data = mycon.find({'ModelName': model_name})

    for i in data:
        json_data = i

    pickled_model = json_data['ModelContent']
    return pickle.loads(pickled_model)


details = save_model_to_db(model=model, client=CLIENT, db=DB, dbconn=DB2CONNECT,
                           model_name='Test1')

print(f"Operation was: {'Successful' if details is not None else 'Failed'}")

y_before_saving = model.predict(X)

print("Loading Model from DB and performing Consistency Check\n")

### Delete Model from Memory
del model

new_model = load_saved_model_from_db(model_name='Test1', client=CLIENT, db=DB, dbconn=DB2CONNECT)

y_after_saving = new_model.predict(X)

assert (y_before_saving == y_after_saving).all()
print("Results before and after Retrieving Model from MongoDB are the same!\n")
