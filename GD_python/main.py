import matplotlib.pyplot as plt
import numpy as np
import ctypes
import itertools


# %%
class MLP_model():

    def __init__(self, npl):
        self.dll = self.load_dll("models/mlp_model.dll")
        self.model = self.create_mlp_model(npl)

    # make an other constructor which loads the model from a file
    # def __init__(self, model_name, a):
    #     self.dll = self.load_dll("models/mlp_model.dll")
    #     self.load_dll(model_name)

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
                                             ctypes.c_int32, flattened_Y_type, ctypes.c_int32, ctypes.c_int32,
                                             ctypes.c_float, ctypes.c_int32, ctypes.c_int32]
        self.dll.train_mlp_model.restype = ctypes.c_void_p

        self.dll.train_mlp_model(self.model, native_flattened_X_array, len(inputs), len(inputs[0]),
                                 native_flattened_Y_array, len(outputs), len(outputs[0]),
                                 learning_rate, epochs, if_classification)

    def predict(self, Xk, nb_outputs):
        predict_X_type = ctypes.c_float * len(Xk)
        native_predict_X = predict_X_type(*Xk)
        self.dll.predict_for_dll.argtypes = [ctypes.c_void_p, predict_X_type, ctypes.c_int32, ctypes.c_int32]
        self.dll.predict_for_dll.restype = ctypes.POINTER(ctypes.c_float)

        rslt = self.dll.predict_for_dll(self.model, native_predict_X, len(Xk), nb_outputs)
        return np.ctypeslib.as_array(rslt, (nb_outputs,))

    def load_dll(self, dll_name):
        """
        Loads dll from file.
        """
        dll = ctypes.cdll.LoadLibrary(dll_name)
        return dll

    def load_model(self, model_name):
        self.dll.load_model.argtypes = [ctypes.c_char_p]
        self.dll.load_model.restype = ctypes.c_void_p

        self.model = self.dll.load_model(model_name)

    def save_model(self, filename):
        string_type = ctypes.c_wchar_p * len(filename)
        print(string_type)

        path = ctypes.c_wchar_p(filename)

        self.dll.save_model.argtypes = [ctypes.c_void_p, string_type]
        self.dll.save_model.restype = None
        self.dll.save_model(self.model, path)

    def test_model(self, X, Y):
        j = 0
        for i in range(Y.shape[0]):
            g = self.predict(X[i].tolist(), 1)
            Yk = Y[i]

            if (g[0] > 0 and Yk[0] > 0) or (g[0] < 0 and Yk[0] < 0):
                j = j + 1

        return j / Y.shape[0]

    def test_model2(self, X, Y, nb_outputs):
        j = 0
        for i in range(Y.shape[0]):
            g = self.predict(X[i].tolist(), nb_outputs)
            Yk = Y[i]
            g_indices = [index for index, item in enumerate(g) if item == max(g)]
            Yk_indices = [index for index, item in enumerate(Yk) if item == max(Yk)]
            if g_indices[0] == Yk_indices[0]:
                j = j + 1
        return j / Y.shape[0]

    def display_limit(self, nb_points, max_value):
        validation_points = [[[i / max_value, j / max_value] for i in range(-nb_points, nb_points)] for j in
                             range(-nb_points, nb_points)]
        XOnes = [p[0] for lp in validation_points for p in lp]
        XTwos = [p[1] for lp in validation_points for p in lp]
        colors = ['blue' if self.predict(p, 1) >= 0 else 'red' for lp in validation_points for p in lp]
        plt.scatter(XOnes, XTwos, c=colors)
        plt.show()


# %%
### Cross :
# Linear Model    : KO
# MLP (2, 1)   : OK
# X = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
# Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

# %%
# plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
# plt.scatter(X[50:100,0], X[50:100,1], color='red')
# plt.show()
# plt.clf()
# %%
X = np.random.random((1000, 2)) * 2.0 - 1.0
Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
    p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in X])

plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
            np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
            color='blue')
plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
            np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1], color='red')
plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
            np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
            color='green')
# plt.show()
# plt.clf()

My_mlp = MLP_model([2, 16, 12, 16, 3])
# %%
print(My_mlp.test_model2(X, Y, 3))
# %%
# My_mlp.display_limit(100, 100)
# %%
My_mlp.train(X.tolist(), Y.tolist(), 10000, 0.1, 1)

My_mlp.save_model("model.txt")

# My_mlp.display_limit(100, 100)
# print(My_mlp.test_model2(X, Y,3))
# My_mlp.train(X.tolist(), Y.tolist(), 100000, 0.01, 1)
# print(My_mlp.test_model2(X, Y,3))
# My_mlp.train(X.tolist(), Y.tolist(), 10000000, 0.0001, 1)
# # My_mlp.display_limit(100, 100)
# print(My_mlp.test_model2(X, Y, 3))
#
# My_mlp.display_limit(100, 10)
# print(My_mlp.test_model(X, Y))
