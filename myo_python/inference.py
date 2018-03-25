# coding: utf-8
import pdb


class EstimatorTF(object):
    def __init__(self):
        pass

    def predict(self):
        pass


def k_test():
    from utils.data_io import DataManager
    from keras.models import load_model

    data_mg = DataManager('./data/10hz', separate_rate=0)
    test_data, _ = data_mg.get_all_data()
    ts_kinematic, ts_emg, ts_target = test_data

    pdb.set_trace()

    model = load_model('./exp/multi2one/rnn_best.h5')
    result = model.predict([ts_kinematic, ts_emg], batch_size=1, verbose=1)
    pdb.set_trace()


if __name__ == "__main__":
    k_test()
