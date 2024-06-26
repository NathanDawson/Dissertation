import os
import pandas as pd


def store_results(results_data):
    """
    Stores the results of model evaluations, including performance metrics, selected features,
    best parameters, and the feature selection technique used, into a pickle file.

    If the results file already exists, new results are appended to it.

    returns:
    None
        The function does not return a value but writes the results to a pickle file.
    """
    results_file_path = 'data/results/final_pipeline_results.pkl.bz2'
    results_df = pd.DataFrame.from_dict(results_data, orient='index')

    if os.path.exists(results_file_path):
        # File exists, read existing data and append new data
        existing_results_df = pd.read_pickle(results_file_path, compression='bz2')
        combined_results_df = pd.concat([existing_results_df, results_df], ignore_index=True)
        combined_results_df.to_pickle(results_file_path, compression='bz2')
    else:
        # File doesn't exist, just use new data
        results_df.to_pickle(results_file_path, compression='bz2')

    print(results_data)
    print("Results stored successfully.")
