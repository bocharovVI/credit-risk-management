import pandas as pd

from preprocess import preprocess_data
from Ensemble import EnsembleNet


def main():
    # Prepare data
    preprocess_data(path_to_dataset='../data', max_categories=30)

    # Initialize model
    ensemble = EnsembleNet(learning_rate=1e-3, batch_size=1024, num_epochs=10)

    # Train model
    ensemble.fit(path_train='../data/processed_data/train', classes_weights=True)

    # Save model
    ensemble.save_model(path_to_model='../data/ensemble_models/with_weights')

    # Take short sample
    data = pd.read_parquet('../data/processed_data/test/test.parquet').sample(100)

    # Make prediction
    y_pred = ensemble.predict(data.drop(['id', 'flag'], axis=1))
    print(y_pred)

    return y_pred


if __name__ == '__main__':
    main()
