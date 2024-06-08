"""pip install catboost"""

from catboost import Pool, CatBoostClassifier
import numpy as np

from optimisation.model.surrogate import Surrogate


class CatBoostClassification(Surrogate):

    def __init__(self, n_dim, l_b, u_b, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        """ 
        Parameters:
        
        """
        self.verbose = False

        self._mid = 0.5
        self._range = 0.5

        # 'MultiClass': multi-class classification
        if 'loss_func' in kwargs:
            self.loss_func = kwargs['loss_func']
        else:
            self.loss_func = 'MultiClass'

        if 'class_weights' in kwargs:
            self.class_weights = kwargs['class_weights']
        else:
            self.class_weights = None

        if 'task_type' in kwargs:
            self.task_type = kwargs['task_type']
        else:
            self.task_type = 'CPU'

        if 'l2_leaf_reg' in kwargs:
            self.l2_leaf_reg = kwargs['l2_leaf_reg']
        else:
            self.l2_leaf_reg = 3.0

        if 'min_data_in_leaf' in kwargs:
            self.min_data_in_leaf = kwargs['min_data_in_leaf']
        else:
            self.min_data_in_leaf = 1

        if 'random_strength' in kwargs:
            self.random_strength = kwargs['random_strength']
        else:
            self.random_strength = 1.0

        if 'border_count' in kwargs:
            self.border_count = kwargs['border_count']
        else:
            self.border_count = 100

        if 'colsample_bylevel' in kwargs:
            self.colsample_bylevel = kwargs['colsample_bylevel']
        else:
            self.colsample_bylevel = 0.65

        if 'depth' in kwargs:
            self.depth = kwargs['depth']
        else:
            self.depth = 10

        if 'lr' in kwargs:
            self.lr = kwargs['lr']
        else:
            self.lr = 0.15

        if 'epoch' in kwargs:
            self.epoch = kwargs['epoch']
        else:
            self.epoch = 100

        self.model = CatBoostClassifier(iterations=self.epoch,
                                        random_strength=self.random_strength,
                                        learning_rate=self.lr,
                                        depth=self.depth,
                                        l2_leaf_reg=self.l2_leaf_reg,
                                        min_data_in_leaf=self.min_data_in_leaf,
                                        colsample_bylevel=self.colsample_bylevel,
                                        border_count=self.border_count,
                                        loss_function=self.loss_func,
                                        task_type=self.task_type,
                                        verbose=self.verbose,
                                        )

    def _train(self):

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        self._x = (self._x - self._mid) / self._range

        # Create Pool
        if self.class_weights is not None:
            train_df = Pool(data=self._x, label=self.y, weight=self.class_weights)
        else:
            train_df = Pool(data=self._x, label=self.y)

        # Train model
        self.model.fit(train_df)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        _x = (_x - self._mid) / self._range

        # Create Pool
        predict_df = Pool(_x)

        # Predict labels
        y = self.model.predict(predict_df)

        return y.flatten()

    def _predict_proba(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        _x = (_x - self._mid) / self._range

        # Create Pool
        predict_df = Pool(_x)

        # Predict function values & re-scale
        y = self.model.predict_proba(predict_df)

        return y

    def _cv_predict(self, model, model_y, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        y = model.predict_proba(_x)

        return y

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for CatBoost!')

    def update_cv_models(self):

        raise Exception('CV models not implemented for CatBoost!')
