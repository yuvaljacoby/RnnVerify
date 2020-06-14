import os
import pickle
import sys

from tqdm import tqdm

IN_SHAPE = (40,)
BASE_FOLDER = "../.."
MODELS_FOLDER = os.path.join(BASE_FOLDER, "models/")
POINTS_PATH = os.path.join(BASE_FOLDER, "models/points.pkl") 


def generate_points(models_folder, number=500, max_t=20):
    mean = 0
    var = 3
    points = []
    models = ['model_classes20_1rnn8_1_32_4.h5', 'model_20classes_rnn2_fc32_epochs200.h5',
              'model_20classes_rnn4_fc32_epochs40.h5', ]
    # ,'model_classes20_1rnn2_0_64_4.h5', 'model_20classes_rnn4_fc32_epochs100.h5'] # os.listdir(models_folder)
    pbar = tqdm(total=number)
    while len(points) <= number:
        fail = 0
        candidate = np.random.normal(mean, var, IN_SHAPE)
        for file in models:
            if os.path.isfile(os.path.join(models_folder, file)):
                try:
                    y_idx_max, other_idx = get_out_idx(candidate, max_t, os.path.join(models_folder, file))
                    if y_idx_max is None or other_idx is None and y_idx_max == other_idx:
                        fail = True
                        continue
                except ValueError:
                    # model with different input shape, it does not matter
                    pass
        if not fail:
            points.append(candidate)
            pbar.update(1)
    pbar.close()

    pickle.dump(points, open(POINTS_PATH, "wb"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = MODELS_FOLDER

    generate_points(path)
