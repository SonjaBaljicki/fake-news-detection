from flatbuffers.builder import np
from tensorflow import keras
from keras import models,saving
from train_cnn import split_dataset
from keras.src.optimizers import Adam,Adadelta



def evaluate_model(model, X_test, Y_test):
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(scores)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    fake_cnt, true_cnt, fake_correct, true_correct = 0, 0, 0, 0
    for x in range(len(X_test)):

        result = model.predict(X_test[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=0)[0]

        if np.around(result) == np.around(Y_test[x]):
            if np.around(Y_test[x]) == 0:
                true_correct += 1
            else:
                fake_correct += 1

        if np.around(Y_test[x]) == 0:
            true_cnt += 1
        else:
            fake_cnt += 1
    print("Fake news accuracy\t: ", round(fake_correct / fake_cnt * 100, 3), "%")
    print("True news accuracy\t: ", round(true_correct / true_cnt * 100, 3), "%")

if __name__ == '__main__':
    X_train, X_test, X_val, Y_train, Y_test, Y_val = split_dataset()

    model = models.load_model("data/model/model.keras")
    opt = Adam(learning_rate=0.001) 
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.load_weights("data/weights.weights.h5")

    evaluate_model(model, X_test, Y_test)