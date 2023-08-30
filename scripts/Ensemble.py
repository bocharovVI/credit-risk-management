import os

import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class ClassificationNet(nn.Module):
    def __init__(self, feature_input):
        super().__init__()

        self.hidden1 = nn.Linear(feature_input, 256)
        self.f1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(256, 10)
        self.f2 = nn.Sigmoid()
        self.output = nn.Linear(10, 1)
        self.f3 = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(self.hidden1(x))
        x = self.f2(self.hidden2(x))
        x = self.f3(self.output(x))

        return x


class EnsembleNet(object):
    """
    Ансамблевая модель машинного обучения построенная на основе моделей
    нейронных сетей из пакета PyTorch. Каждая модель обучается на
    отдельном чанке с предобработанными данными.

    Parameters
    ----------
    learning_rate : float, default=1e-3
        Множитель перед градиентом.

    batch_size : int, default=1024
        Определяет размер батча класса DataLoader.

    num_epochs : int, default=10
        Число эпох обучения.

    train_size : float, default=0.8
        Доля обучающей выборки от датафрейма.

    verbose : bool, default=True
        Вывод подробной информации в процессе обучения.

    Attributes
    ----------

    models : list
        Список обученных моделей.

    features : list
        Хранит список признаков для каждой модели.

    loss_test : list
        Список со списком значений функции потерь
        для каждой модели.

    roc_auc_test : list
        Список со списком значений метрики
        для каждой модели.
    """

    def __init__(
            self,
            learning_rate=1e-3,
            batch_size=1024,
            num_epochs=10,
            train_size=0.8,
            verbose=True
    ):
        self.models = []
        self.features = []
        self.loss_test = []
        self.roc_auc_test = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_size = train_size
        self.verbose = verbose

    def fit(self, path_train, classes_weights=False):
        """
        Для каждого файла с данными в path_train обучает одну модель
        и записывает в список моделей self.models.

        Parameters
        ----------
        path_train : str
            Путь до директории с обучающей выборкой.

        classes_weights : bool, default=False
            Флаг для весов классов.

        Returns
        -------
        self
            Обученная ансамблевая модель.
        """

        self.models = []
        self.features = []
        self.loss_test = []
        self.roc_auc_test = []

        chunk_paths = sorted([os.path.join(path_train, filename) for filename in os.listdir(path_train)])
        for i, chunk in enumerate(chunk_paths):
            # объявим список ошибок и для текущей модели
            loss_test = []
            roc_auc_test = []

            # загружаем данные и разделяем на трейн и тест
            data = pd.read_parquet(chunk)
            X = data.drop(['id', 'flag'], axis=1).to_numpy()
            y = data['flag'].to_numpy()
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=self.train_size, random_state=12,
                                                            stratify=y)
            self.features.append(list(data.drop(['id', 'flag'], axis=1).columns))
            del data

            # считаем веса классов
            class_weights = None
            if classes_weights:
                class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                                  classes=np.unique(ytrain),
                                                                  y=ytrain)
                class_weights = torch.tensor(class_weights, dtype=torch.float)

            # выбираем вычислительный блок
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # объявляем датасет
            train_dataset = MyDataset(Xtrain, ytrain)
            Xtest_tensor = torch.from_numpy(Xtest.astype(np.float32)).to(device)
            ytest_tensor = torch.from_numpy(ytest.astype(np.float32)).to(device)

            # объявляем даталоадер
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

            # инициализируем модель
            self.models.append(ClassificationNet(feature_input=X.shape[1]))
            self.models[-1].to(device)
            optimizer = torch.optim.Adam(self.models[-1].parameters(), lr=self.learning_rate)
            if self.verbose:
                print(f'Start train model {i + 1}')

            for epoch in range(self.num_epochs):
                for X, y in train_dataloader:

                    # отправляем батчи на cpu или gpu
                    X, y = X.to(device), y.to(device)

                    # делаем предикт
                    pred = self.models[-1](X)

                    # вычисляем веса классов
                    if classes_weights:
                        weights = torch.zeros_like(y.unsqueeze(-1))
                        weights[y == 0] = class_weights[0]
                        weights[y == 1] = class_weights[1]
                    else:
                        weights = None

                    # делаем шаг градиентного спуска
                    loss = F.binary_cross_entropy(pred, y.unsqueeze(-1), weight=weights)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    # вычисляем веса классов
                    if classes_weights:
                        weights = torch.zeros_like(ytest_tensor.unsqueeze(-1))
                        weights[ytest_tensor == 0] = class_weights[0]
                        weights[ytest_tensor == 1] = class_weights[1]
                    else:
                        weights = None

                    # считаем потери на тесте
                    loss = F.binary_cross_entropy(self.models[-1](Xtest_tensor), ytest_tensor.unsqueeze(-1),
                                                  weight=weights).item()

                    # вычисляем значение метрики auc
                    fpr, tpr, _ = roc_curve(ytest, self.models[-1](Xtest_tensor).cpu().detach().numpy().ravel())
                    roc_auc = auc(fpr, tpr)
                    loss_test.append(loss)
                    roc_auc_test.append(roc_auc)
                    if self.verbose:
                        print(f'epoch {epoch + 1:3.0f} | loss {loss:0.5f} | roc_auc {roc_auc:0.5f}')

            self.loss_test.append(loss_test)
            self.roc_auc_test.append(roc_auc_test)
            if self.verbose:
                print(f'Finish train model {i + 1}\n')

    def predict(self, X, method='mean'):
        """
        Принимает на вход матрицу объект-признак и
        делает предсказание классов указанным методом.

        Parameters
        ----------
        X : array-like, sparse matrix of shape (n_samples, n_features)
            Закодированная матрица объект-признак.

        method : {'mean', 'vote'}, default='mean'
            Метод усреднения предсказания:

            - 'mean': вычисляется среднее выходов
                      всех моделей и округляется;
            - 'vote': выбирается мода округленных
                      выходов всех моделей;

        Returns
        -------
        T : array-like of shape (n_samples,)
            Метод возвращает вектор предсказания классов.
        """

        y_list = []

        if len(self.models) == 0:
            raise BaseException('no models were fitted')

        for features, model in zip(self.features, self.models):
            X = pd.DataFrame(X, columns=features).fillna(0.)
            X_tensor = torch.from_numpy(X.to_numpy().astype(np.float32))
            y = model(X_tensor).detach().numpy().ravel()
            y_list.append(y)

        y = np.array(y_list).T

        if method == 'mean':
            return np.round(pd.DataFrame(y).mean(axis=1).to_numpy())
        elif method == 'vote':
            return pd.DataFrame(np.round(y)).mode(axis=1)[0].squeeze().to_numpy()
        else:
            raise BaseException('unknown method given')

    def predict_proba(self, X):
        """
        Принимает на вход матрицу объект-признак и
        делает предсказание вероятности принадлежности
        к положительному классу.

        Parameters
        ----------
        X : array-like, sparse matrix of shape (n_samples, n_features)
            Матрица объект-признак.

        Returns
        -------
        T : array-like of shape (n_samples,)
            Метод возвращает вектор вероятностей принадлежности
            к положительному классу.
        """

        y_list = []

        if len(self.models) == 0:
            raise BaseException('no models were fitted')

        for features, model in zip(self.features, self.models):
            X = pd.DataFrame(X, columns=features).fillna(0.)
            X_tensor = torch.from_numpy(X.to_numpy().astype(np.float32))
            y = model(X_tensor).detach().numpy().ravel()
            y_list.append(y)

        y = np.array(y_list).T

        return pd.DataFrame(y).mean(axis=1).to_numpy()

    def save_model(self, path_to_model='../data/ensemble_models/'):
        """
        Сохраняет модель и ее параметры в указанной директории.
        Если директория не существует, то создает ее.

        Parameters
        ----------
        path_to_model : str, default='../data/ensemble_models/'
            Путь до модели.

        Return
        ------
        model : file .pt
            Файл с обученной моделью.
        """

        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)

        torch.save({model: (model.state_dict(), features) for model, features in zip(self.models, self.features)},
                   path_to_model + f'model_{datetime.now().strftime("%Y%m%d%H%M")}.pt')

    def load_model(self, path_to_model):
        """
        Загружает модель и ее параметры из указанной директории.

        Parameters
        ----------
        path_to_model : str
            Путь до модели.

        Return
        ------
        self : EnsembleNet
            Сконфигурированная модель
        """

        checkpoint = torch.load(path_to_model)
        for model in checkpoint:
            model.load_state_dict(checkpoint[model][0])
            model.eval()
            self.models.append(model)
            self.features.append(checkpoint[model][1])

        return self