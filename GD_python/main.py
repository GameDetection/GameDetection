import matplotlib.pyplot as plt
import numpy as np
import ctypes
import itertools


# %%
class MLP_model():

    def __init__(self, npl):
        self.model = None
        self.dll = self.load_dll("models/mlp_model.dll")
        self.model = self.create_mlp_model(npl)

    def create_mlp_model(self, npl):
        """
        Creates mlp model.
        """
        npl_type = ctypes.c_int32 * len(npl)
        self.dll.create_mlp_model.argtypes = [npl_type, ctypes.c_int32]
        self.dll.create_mlp_model.restype = ctypes.c_void_p

        self.dll.delete_mlp_model.argtypes = [ctypes.c_void_p]
        self.dll.delete_mlp_model.restype = None

        npl_type = ctypes.c_int32 * len(npl)
        native_npl_array = npl_type(*npl)

        self.model = self.dll.create_mlp_model(native_npl_array, len(npl))

        return self.model

    def delete_mlp_model(self):
        """
        Deletes mlp model.
        """
        self.dll.delete_mlp_model(self.model)

    def train(self, inputs, outputs, epochs, learning_rate, if_classification):
        """
        Trains mlp model.
        """
        flattened_X = list(itertools.chain(*inputs))
        flattened_Y = list(itertools.chain(*outputs))
        flattened_X_type = ctypes.c_float * len(flattened_X)
        native_flattened_X_array = flattened_X_type(*flattened_X)

        flattened_Y_type = ctypes.c_float * len(flattened_Y)
        native_flattened_Y_array = flattened_Y_type(*flattened_Y)

        self.dll.train_mlp_model.argtypes = [ctypes.c_void_p, flattened_X_type, ctypes.c_int32,
                                             ctypes.c_int32, flattened_Y_type, ctypes.c_int32, ctypes.c_float,
                                             ctypes.c_int32,
                                             ctypes.c_int32]
        self.dll.train_mlp_model.restype = ctypes.c_void_p

        self.dll.train_mlp_model(self.model, native_flattened_X_array, len(inputs), len(inputs[0]),
                                 native_flattened_Y_array, len(outputs), learning_rate, epochs, if_classification)

    def predict(self, Xk, nb_outputs):
        predict_X_type = ctypes.c_float * len(Xk)
        native_predict_X = predict_X_type(*Xk)
        self.dll.predict_for_dll.argtypes = [ctypes.c_void_p, predict_X_type, ctypes.c_int32, ctypes.c_int32]
        self.dll.predict_for_dll.restype = ctypes.POINTER(ctypes.c_float)

        rslt = self.dll.predict_for_dll(self.model, native_predict_X, len(Xk), 1)

        return np.ctypeslib.as_array(rslt, (1,))

    def load_dll(self, dll_name):
        """
        Loads dll from file.
        """
        dll = ctypes.cdll.LoadLibrary(dll_name)
        return dll

    def test_model(self, X, Y):
        j = 0
        for i in range(Y.shape[0]):
            g = self.predict(X[i].tolist(), 1)
            Yk = Y[i]

            if (g[0] > 0 and Yk[0] > 0) or (g[0] < 0 and Yk[0] < 0):
                j = j + 1

        return j / Y.shape[0]

    def display_limit(self, nb_points, max_value):
        validation_points = [[[i / max_value, j / max_value] for i in range(-nb_points, nb_points)] for j in range(-nb_points, nb_points)]
        XOnes = [p[0] for lp in validation_points for p in lp]
        XTwos = [p[1] for lp in validation_points for p in lp]
        colors = ['blue' if self.predict(p, 1) >= 0 else 'red' for lp in validation_points for p in lp]
        plt.scatter(XOnes, XTwos, c=colors)
        plt.show()
# %%
### Cross :
# Linear Model    : KO
# MLP (2, 1)   : OK
X = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

#%%
plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
plt.scatter(X[50:100,0], X[50:100,1], color='red')
plt.show()
plt.clf()
# %%
My_mlp = MLP_model([2, 3, 4, 2, 1])
# %%
print(My_mlp.test_model(X, Y))
# %%
My_mlp.display_limit(100, 100)
# %%
My_mlp.train(X.tolist(), Y.tolist(), 1000000, 0.01, 1)

My_mlp.display_limit(100, 100)
print(My_mlp.test_model(X, Y))