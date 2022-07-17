import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, data_full, weighting=True):

        # Топ покупок каждого юзера
        self.top_purchases = data_full.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        # self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data_full.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        # self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)

        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        self.data_ = data
        self.item_fact = self.model.item_factors
        self.user_fact = self.model.user_factors

        self.users_to_pred = data['user_id'].unique()

    @staticmethod
    def _prepare_matrix(data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    # @staticmethod
    def fit_2_level(self, item_features_st2, user_features_st2, recomender_my='own', N=5, N_inpt=200):

        data_train_st2 = self.data_

        if recomender_my == 'own':
            N_inpt = 200  # при таком соотношении датасет для обучения на второй стадии сбалансирован
            data_stage = self._get_recommendations(model=self.own_recommender, N=N_inpt)
        elif recomender_my == 'als':
            N_inpt = 50
            data_stage = self._get_recommendations(model=self.model, N=N_inpt)
        else:
            data_stage = {}
            for i in range(len(self.users_to_pred)):
                data_stage[self.id_to_userid[i]] = self._extend_with_top_popular(recommendations=[], N=N_inpt)

        # формируем датафрейм по кандидатам(с первого уровня модели)
        temp = pd.DataFrame(columns=['user_id', 'item_id'])
        temp['user_id'] = data_stage.keys()
        temp['item_id'] = data_stage.values()

        # создаем series где индекс это порядок по user_id а значения это предложенные товары
        data_stage_2 = temp.copy()
        s = data_stage_2.apply(lambda x: pd.Series(x['item_id']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'

        # создаем датафрейм на основе предыдущего сериес с user id и флагом покупки.
        data_stage_2 = data_stage_2.drop('item_id', axis=1).join(s)
        data_stage_2['flag'] = 1

        # создаем датафрейм покупок(истинный класс)
        data_targets = data_train_st2[['user_id', 'item_id']].copy()
        data_targets['target'] = 1  # тут только покупки

        # объединяем датафреймы, тем самым получая истинные метки класса
        data_targets = data_stage_2.merge(data_targets, on=['user_id', 'item_id'], how='left')

        # заполняем вторую метку класса и удаляем вспомогательный столбец
        data_targets['target'].fillna(0, inplace=True)
        data_targets.drop('flag', axis=1, inplace=True)

        # добавляем фичи пользователей и фичи юзверей
        data_targets = data_targets.merge(item_features_st2, on='item_id', how='left')
        data_targets = data_targets.merge(user_features_st2, on='user_id', how='left')

        # создаем финальный датасет
        X_train = data_targets.drop('target', axis=1)
        y_train = data_targets[['target']]

        print(data_targets['target'].mean())

        cat_feats = X_train.columns[2:].tolist()
        X_train[cat_feats] = X_train[cat_feats].astype('category')

        ''' 'boosting_type': 'gbdt',
         'max_depth': -1,
         'objective': 'binary',
         'num_leaves': 17,
         'learning_rate': 0.01,
         'max_bin': 512,
         'subsample_for_bin': 200,
         'subsample': 0.75,
         'subsample_freq': 1,
         'colsample_bytree': 0.8,
         'reg_alpha': 6,
         'reg_lambda': 7,
         'min_split_gain': 0.5,
         'min_child_weight': 1,
         'min_child_samples': 5,
         'scale_pos_weight': 1,
         'num_class': 1,
         'metric': 'binary_error',
         'verbosity': -1}'''

        lgb = LGBMClassifier(boosting_type='gbdt',
                             min_split_gain=0.5,
                             min_child_weight=1,
                             min_child_samples=5,
                             scale_pos_weight=1,
                             num_class=1,
                             metric='binary_error',
                             verbosity=-1,
                             max_bin=512,
                             subsample_for_bin=200,
                             subsample_freq=1,
                             colsample_bytree=0.8,
                             learning_rate=0.01,
                             n_estimators=25,
                             num_leaves=17,
                             objective='binary',
                             reg_alpha=6,
                             reg_lambda=7,
                             seed=500,
                             subsample=0.75,
                             categorical_column=cat_feats)

        self.lgb = lgb.fit(X_train, y_train)

        # предсказываем

        # train_preds = lgb.predict(X_train)

        ''' 


        #Сортировка
        #indices = (-train_preds).argpartition(N, axis=None)[:N]
        #users_lvl_2['candidates200'].loc[users_ids] = data_train_lvl_1['item_id'].unique()[indices]
        data_predict = data_targets.copy()
        data_predict['probobility'] = train_preds
        data_predict = data_predict.sort_values(by='probobility', ascending=False)

        finish_df = pd.DataFrame()
        for el in data_predict['user_id'].unique():
            k = data_predict[data_predict['user_id'] == el].sort_values(by='probobility', ascending=False)[:N].groupby('user_id')['item_id'].unique().reset_index()   
            finish_df = pd.concat([finish_df, k], ignore_index=True)

        finish_df.columns=['user_id', 'item_id']

        for i in range(finish_df.shape[0]):
            finish_df['item_id'][i] = self._extend_with_top_popular(finish_df['item_id'][i].tolist(), N=N)'''

        # return finish_df

    # так как нормальный результат показал только item2item, то и предсказывать будем его.
    def predict_2_level(self, users_to_pred, item_features_st2, user_features_st2, N=5, N_inpt=30):

        data_stage = self._get_recommendations(model=self.own_recommender, N=N_inpt)

        # формируем новый словарь для предсказаний
        data_stage_temp = {}
        for i in users_to_pred:
            if i not in data_stage.keys():
                data_stage_temp[i] = self._extend_with_top_popular([], N=N_inpt)
            else:
                data_stage_temp[i] = data_stage[i]
        del (data_stage)

        # формируем датафрейм по кандидатам(с первого уровня модели)
        temp = pd.DataFrame(columns=['user_id', 'item_id'])
        temp['user_id'] = data_stage_temp.keys()
        temp['item_id'] = data_stage_temp.values()

        # создаем series где индекс это порядок по user_id а значения это предложенные товары
        data_stage_2 = temp.copy()
        s = data_stage_2.apply(lambda x: pd.Series(x['item_id']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'

        # создаем датафрейм на основе предыдущего сериес с user id
        data_stage_2 = data_stage_2.drop('item_id', axis=1).join(s)

        # добавляем фичи пользователей и фичи юзверей
        data_stage_2 = data_stage_2.merge(item_features_st2, on='item_id', how='left')
        X_ = data_stage_2.merge(user_features_st2, on='user_id', how='left')

        cat_feats = X_.columns[2:].tolist()
        X_[cat_feats] = X_[cat_feats].astype('category')

        # предсказываем
        train_preds = self.lgb.predict(X_)

        X_['pred'] = train_preds
        df_preds = X_[X_['pred'] == 1].groupby('user_id')['item_id'].unique().reset_index()

        # Создаем словарь и проверяем колличество ответов
        preds_dict = {}
        for i in X_[X_['pred'] == 1].groupby('user_id')['item_id'].unique().reset_index().iloc:
            if len(i['item_id']) > N:
                recs = np.random.choice(i['item_id'], size=N, replace=False)
                preds_dict[i['user_id']] = recs.tolist()
            elif len(i['item_id']) < N:
                preds_dict[i['user_id']] = self._extend_with_top_popular(i['item_id'].tolist(), N=N)

        return preds_dict

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=50, regularization=0.001, iterations=40, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, model, user=False, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res, weights = model.recommend(userid=self.users_to_pred,
                                       user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                       N=N,
                                       filter_already_liked_items=False,
                                       recalculate_user=True)

        # res = [self.id_to_itemid[k][i] for i in k for k in res]
        fin_res = []
        for i in range(len(res)):
            fin_res_ = []
            for j in range(len(res[i])):
                try:
                    fin_res_.append(self.id_to_itemid[res[i, j]])
                except:
                    c = 1 + 1  # print(res[i,j])
            fin_res.append(fin_res_)
            # распарсить индексы в числа

        if N > 50:
            N = int(N * 1.1)

        user_dict = {}
        if user:
            local_id = self.userid_to_id[user]
            fin_res = fin_res[local_id]
            fin_res = self._extend_with_top_popular(fin_res, N=N)
            user_dict[user] = fin_res

            assert len(fin_res) == N, 'Количество рекомендаций != {}'.format(N)

        else:
            for i in range(len(fin_res)):
                fin_res[i] = self._extend_with_top_popular(fin_res[i], N=N)
                user_dict[self.id_to_userid[i]] = fin_res[i]

        return user_dict

    def get_als_recommendations(self, user=False, N=5):
        """Рекомендации через стардартные библиотеки implicit"""
        if user:
            self._update_dict(user_id=user)
        return self._get_recommendations(model=self.model, user=user, N=N)

    def get_own_recommendations(self, user=False, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        if user:
            self._update_dict(user_id=user)
        return self._get_recommendations(model=self.own_recommender, user=user, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        self._update_dict(user_id=user)
        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []
        self._update_dict(user_id=user)

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]  # удалим юзера из запроса

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res