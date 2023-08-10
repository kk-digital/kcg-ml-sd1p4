import zipfile
import os
import json
import numpy as np
from pprint import pprint
import argparse
import sys
import datetime
import sklearn.metrics
import autosklearn.regression
import matplotlib.pyplot as plt
import autosklearn
# print autosklearn version
print('autosklearn: %s' % autosklearn.__version__)


def load_dataset_generated_from_random_prompts(dataset_zip_path):
    # load zip
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        data_list = []
        file_paths = zip_ref.namelist()
        for file_path in file_paths:

            file_extension = os.path.splitext(file_path)[1]
            if file_extension == ".jpg":
                # get filename
                file_path_no_extension = os.path.splitext(file_path)[0]

                file_path_json = file_path_no_extension + ".json"
                with zip_ref.open(file_path_json) as file:
                    json_content = json.load(file)

                file_path_embedding = file_path_no_extension + ".embedding.npz"
                with zip_ref.open(file_path_embedding) as file:
                    embedding = np.load(file)
                    embedding_data = embedding['data']

                file_path_clip = file_path_no_extension + ".clip.npz"
                with zip_ref.open(file_path_clip) as file:
                    clip = np.load(file)
                    clip_data = clip['data']

                file_path_latent = file_path_no_extension + ".latent.npz"
                with zip_ref.open(file_path_latent) as file:
                    latent = np.load(file)
                    latent_data = latent['data']

                data_list.append({"json": json_content, "embedding": embedding_data, "clip": clip_data, "latent": latent_data})
        return data_list

def split_and_process_dataset(data_list, X_input):
    print("Length of dataset: {0}".format(len(data_list)))

    # define dataset
    # use 70% for train, 30% for test
    num_train = int(len(data_list) * 0.7)

    # X - clip or embedding
    # y - chad score
    X_value = X_input
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(data_list)):
        if len(X_train) < num_train:
            # add to X_train and y_train
            val = data_list[i][X_value][0]

            if X_value == "embedding":
                val = val.flatten()

            X_train.append(val)
            y_train.append(data_list[i]['json']['chad_score'])
        else:
            # add to X_test and y_test
            val = data_list[i][X_value][0]
            if X_value == "embedding":
                val = val.flatten()

            X_test.append(val)
            y_test.append(data_list[i]['json']['chad_score'])
    return X_train, y_train, X_test, y_test


def run_auto_ml(X_train, y_train, X_test, y_test, automl_total_time, automl_per_run_time, output_path):
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=automl_total_time,
        memory_limit=102400,
        per_run_time_limit=automl_per_run_time,
        tmp_folder="./tmp/autosklearn_regression_example_tmp",
    )
    automl.fit(X_train, y_train, dataset_name="test-dataset")

    print("--------------------------------------")
    print(automl.leaderboard())
    print("--------------------------------------")
    pprint(automl.show_models(), indent=4)
    print("--------------------------------------")
    train_predictions = automl.predict(X_train)
    print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
    test_predictions = automl.predict(X_test)
    print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

    plt.scatter(train_predictions, y_train, label="Train samples", c="#d95f02")
    plt.scatter(test_predictions, y_test, label="Test samples", c="#7570b3")
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.legend()
    plt.plot([-60, 60], [-60, 60], c="k", zorder=0)
    plt.xlim([-60, 60])
    plt.ylim([-60, 60])
    plt.tight_layout()
    plt.show()

    # save plot
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f'{timestamp}-auto-ml.png'
    full_output_path = os.path.join(output_path, file_name)
    plt.savefig(full_output_path)

def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(description="Run automl on a dataset with embeddings or clip feature and chad score")
    parser.add_argument('--x-input', type=str, default='clip', help='X-input will be either embedding or clip')
    parser.add_argument('--total-time', type=int, default=240, help='Time limit in seconds for the search of appropriate models')
    parser.add_argument('--per-run-time', type=int, default=30, help='Time limit for a single call to the machine learning model')
    parser.add_argument('--dataset-zip-path', type=str, help='Path to the dataset to be used')
    parser.add_argument('--output', type=str, help='Output path where the plot image will be saved')

    return parser.parse_args()

def ensure_required_args(args):
    """Check if required arguments are set."""
    if not args.x_input:
        print('Error: --x-input is required')
        sys.exit(1)
    if not args.dataset_zip_path:
        print('Error: --dataset-zip-path is required')
        sys.exit(1)
    if not args.output:
        print('Error: --output is required')
        sys.exit(1)



if __name__ == "__main__":
    args = parse_arguments()
    ensure_required_args(args)

    dataset_zip_path = args.dataset_zip_path
    output_path = args.output
    X_input = args.x_input
    automl_total_time = args.total_time
    automl_per_run_time = args.per_run_time

    # dataset_zip_path = "./input/set_0002.zip"
    # output_path = "./"
    # X_input = "clip"
    # automl_total_time= 240
    # automl_per_run_time = 30

    data_list = load_dataset_generated_from_random_prompts(dataset_zip_path)
    X_train, y_train, X_test, y_test = split_and_process_dataset(data_list, X_input)
    run_auto_ml(X_train, y_train, X_test, y_test, automl_total_time, automl_per_run_time, output_path)