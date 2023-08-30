import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(path_to_dataset='../data', max_categories=30):
    if not os.path.exists(os.path.join(path_to_dataset, 'processed_data')):
        os.makedirs(os.path.join(path_to_dataset, 'processed_data'))

    total_features = set()
    test = []

    dataset_paths = sorted([os.path.join(path_to_dataset, 'train_data', filename)
                            for filename in os.listdir(os.path.join(path_to_dataset, 'train_data'))
                            if filename.startswith('train')])
    for step, chunk_path in enumerate(dataset_paths):
        print(f'{step}. chunk_path {chunk_path}')

        # прочитать один файл с данными
        transactions_frame = pd.read_parquet(chunk_path, columns=None)

        # удалим дубликаты
        transactions_frame = transactions_frame.drop_duplicates()

        # закодируем данные
        columns_to_encode = transactions_frame.drop(['id', 'rn'], axis=1).columns
        ohe = OneHotEncoder(sparse_output=False, max_categories=max_categories, drop='first')
        encoded_frame = ohe.fit_transform(transactions_frame[columns_to_encode])

        # обновим множество закодированных фичей
        total_features.update(list(ohe.get_feature_names_out()))

        # объединим закодированные категории с id
        data_preprocessed = pd.concat([transactions_frame['id'],
                                       pd.DataFrame(encoded_frame, columns=ohe.get_feature_names_out())], axis=1)
        del transactions_frame, encoded_frame

        # агрегируем по id: суммируем закодированные фичи по каждому клиенту
        data_preprocessed = data_preprocessed.groupby('id', as_index=False).agg('sum')

        # смерджим с целевой переменной flag
        targets = pd.read_csv(os.path.join(path_to_dataset, 'train_target', 'train_target.csv'))
        data_preprocessed = data_preprocessed.merge(targets, how='inner', on='id')
        del targets

        # подготовим код чанка
        block_as_str = str(step)
        if len(block_as_str) == 1:
            block_as_str = '00' + block_as_str
        else:
            block_as_str = '0' + block_as_str

        data_preprocessed_train, data_preprocessed_test = train_test_split(data_preprocessed,
                                                                           train_size=0.8,
                                                                           random_state=12,
                                                                           stratify=data_preprocessed['flag'])

        # сохраним обработанный блок в train и test
        data_preprocessed_train.to_parquet(os.path.join(path_to_dataset,
                                                        f'processed_data/train/processed_chunk_{block_as_str}.parquet'))
        print(f'Saved {path_to_dataset}/processed_data/train/processed_chunk_{block_as_str}.parquet')

        # добавим data_preprocessed_test в список test
        test.append(data_preprocessed_test)

    test_df = pd.concat(test, axis=0).fillna(0.)
    test_df.to_parquet(os.path.join(path_to_dataset, 'processed_data/test/test.parquet'))
    print(f'Saved test.parquet')
