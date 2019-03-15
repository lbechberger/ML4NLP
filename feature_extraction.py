from generation_functions import *
from sklearn.metrics import pairwise
import gensim, os
import sklearn
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, confusion_matrix, make_scorer

class FeatureExtraction:

    def __init__(self, articles, categories, labels):
        self.articles = articles
        self.categories = categories
        self.labels = labels

    def get_category_embeddings(self, embedding=[]):
        """
        Calculate the embedding vectors of the categories
        :param embedding: embedding vector. if not provided, google's word2vec is used
        :return: mean embedding of each category name
        """
        # Embedding of each category
        if os.path.isfile('./data/embed_categories.pickle'):
            embedded_categories = pickle.load(open("./data/embed_categories.pickle", "rb"))
        else:
            if not embedding:
                embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                            binary=True).wv
            # calculate embeddings and take mean
            embedded_categories = []
            for c in self.categories:
                if c == "Wackynews":
                    c = "Wacky News"
                # get the embeddings
                embeds = get_w2v_string(c, embedding)
                # calculate mean if there was more than one word
                if len(embeds) > 1:
                    embeds = [sum(x)/len(x) for x in zip(*embeds)]
                else:
                    embeds = embeds[0]
                # append embeds to list of all category embeddings
                embedded_categories.append(embeds)

        pickle.dump(embedded_categories, open("./data/embed_categories.pickle", "wb"))

        return embedded_categories

    def get_article_embedding(self, embedding=[]):
        """
        Calculate the embeddings of each article
        :param embedding: embedding vector. if not provided, google's word2vec is used
        :return: mean embedding of each articles
        """
        if os.path.isfile('./data/embed_articles.pickle'):
            embedded_articles = pickle.load(open("./data/embed_articles.pickle", "rb"))
        else:
            if not embedding:
                embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                            binary=True).wv
            a_counter = 1
            embedded_articles = []
            for a in self.articles:
                embed = get_w2v_string(a, embedding)

                # append mean embedding of article to list of all article embeddings
                embedded_articles.append([sum(x)/len(x) for x in zip(*embed)])
                print("Article: {} /{}".format(a_counter, len(self.articles)))
                a_counter += 1

            pickle.dump(embedded_articles, open("./data/embed_articles.pickle", "wb"))
        return embedded_articles

    def get_cosine_sim(self):
        """
        Calculate the cosine similarity between the articles' and categories' embedding vectors
        :return: Float array with the similarity measures
        """
        if os.path.isfile('./data/cosine_similarities.pickle'):
            cos_sims = pickle.load(open("./data/cosine_similarities.pickle", "rb"))
        else:
            cos_sims = []
            embedded_articles = self.get_article_embedding()
            embedded_categories = self.get_category_embeddings()
            for a in embedded_articles:
                cs = []
                for c in embedded_categories:
                    cs.append(pairwise.cosine_similarity([a], [c]))
                cos_sims.append(cs)
            pickle.dump(cos_sims, open("./data/cosine_similarities.pickle", "wb"))
        return cos_sims

    def extract_user_similarities(self, users):
        """
        Return max, min, mean similarities of the categories the user liked
        :param user: Array of all user where each user's values is in range [0,1]
        :return: Max, min and mean array containing the respective similarities
        """
        if os.path.isfile('./data/m_similarities.pickle'):
            max_val, min_val, mean_val = pickle.load(open("./data/m_similarities.pickle", "rb"))
            return max_val, min_val, mean_val
        else:
            similarities = self.get_cosine_sim()
            max_val = []
            min_val = []
            mean_val = []
            for idx, user in users.iterrows():
                u_max = []
                u_min = []
                u_mean = []
                for a in range(len(self.articles)):
                    values = []
                    for i in range(len(user)):
                        if user[i] == 1:
                            # append the values to a list where we take the values later on (because of bad format,
                            # have to access them via [0][0]
                            values.append((similarities[a][i])[0][0])
                    u_max.append(np.max(values))
                    u_min.append(np.min(values))
                    u_mean.append(np.mean(values))
                max_val.append(u_max)
                min_val.append(u_min)
                mean_val.append(u_mean)
            pickle.dump([max_val, min_val, mean_val], open("./data/m_similarities.pickle", "wb"))
        return max_val, min_val, mean_val

    def extract_user_keys(self, users, keys):
        if os.path.isfile('./data/m_keys.pickle'):
            max_val, min_val, mean_val = pickle.load(open("./data/m_keys.pickle", "rb"))
            return max_val, min_val, mean_val
        else:
            max_val = []; mean_val = []; min_val = []
            for idx, user in users.iterrows():
                u_max = []; u_mean = []; u_min = []
                for a in range(len(self.articles)):
                    values = []
                    for i in range(len(user)):
                        if user[i] == 1:
                            values.append((keys[a][i]))
                    u_max.append(np.max(values))
                    u_mean.append(np.mean(values))
                    u_min.append(np.min(values))
                max_val.append(u_max)
                mean_val.append(u_mean)
                min_val.append(u_min)
            pickle.dump([max_val, min_val, mean_val], open("./data/m_keys.pickle", "wb"))
        return max_val, min_val, mean_val

    def extract_user_checks(self, users, check):
        if os.path.isfile('./data/m_checks.pickle'):
            max_val, min_val, mean_val = pickle.load(open("./data/m_checks.pickle", "rb"))
            return max_val, min_val, mean_val
        else:
            max_val = []; mean_val = []; min_val = []
            for idx, user in users.iterrows():
                u_max = []; u_mean = []; u_min = []
                for a in range(len(self.articles)):
                    values = []
                    for i in range(len(user)):
                        if user[i] == 1:
                            values.append((check[a][i]))
                    u_max.append(np.max(values))
                    u_mean.append(np.mean(values))
                    u_min.append(np.min(values))
                max_val.append(u_max)
                mean_val.append(u_mean)
                min_val.append(u_min)
            pickle.dump([max_val, min_val, mean_val], open("./data/m_checks.pickle", "wb"))
        return max_val, min_val, mean_val

    def get_article_length(self):
        """
        Calculates the length of each article
        :return: Array with the lengths of each article
        """
        lengths = []
        for a in self.articles:
            lengths.append(len(a))
        return lengths

    def category_check(self):
        """
        Check if the category names appear in each articles
        :return: int array [0,1]
        """
        if os.path.isfile('./data/category_in_articles.pickle'):
            cat_in_articles = pickle.load(open("./data/category_in_articles.pickle", "rb"))
        else:
            cat_in_articles = []
            for a in self.articles:
                a = a.lower()
                bool_array = []
                for name in self.categories:
                    name = name.lower()
                    if name == "wackynews":
                        name = "wacky"  # not news, because news is highly uncorrelated to the category
                    c_in_a = 0
                    for c in name.split(" "):
                        if c == "and":
                            continue
                        if c in a:
                            c_in_a = 1
                            break
                    bool_array.append(c_in_a)
                cat_in_articles.append(bool_array)
            pickle.dump(cat_in_articles, open("./data/category_in_articles.pickle", "wb"))

        return cat_in_articles

    def keyword_check(self, keywords):
        """
        Checks the article for predefined keywords/terms
        :param keywords: txt file with keywords
        :return: array for each categories's keywords in each article[0,1]
        """
        if os.path.isfile('./data/keywords_in_articles.pickle'):
            keys_in_articles = pickle.load(open("./data/keywords_in_articles.pickle", "rb"))
        else:
            keys_in_articles = []
            for a in self.articles:
                a = a.lower()
                l_in_a = []
                for line in keywords:
                    # remove brackets from lines, so they look like: "Term A", "Term B"
                    line = line[line.find("[")+1:line.find("]")]
                    count = 0
                    for w in line.split(", "):  # for each word in line
                        w = w.replace('"', '').lower()  # remove parenthesis from words
                        count += sum(1 for _ in re.finditer(r'%s' % re.escape(w), a))  # count number of appearances
                    l_in_a.append(count)
                keys_in_articles.append(l_in_a)
            pickle.dump(keys_in_articles, open("./data/keywords_in_articles.pickle", "wb"))

        return keys_in_articles

    def get_features(self, users_db):
        """
        Calculates all features and return them as an array
        Features are:
        - cosine similarities of articles and categories based on user preferences (max, min, mean),
        - boolean array of category names occuring in article,
        - the article length
        boolean array of keywords occuring in article
        :param users_db: pandas dataframe of user
        :return: array of size 4500, with all the features for each user
        """
        # calculate the cosine similarities between articles and categories based on user preferences
        maximum_sim, minimum_sim, mean_sim = self.extract_user_similarities(users_db)
        # check if the category names appear in text and use appearances based on user preferences
        check = self.category_check()
        # max_check, min_check, mean_check = self.extract_user_checks(users_db, check)
        # get the length of each article
        article_lengths = self.get_article_length()
        # check if the specified keywords appear in text
        lines = [line.rstrip('\n') for line in open("./data/keywords.txt")]
        keys = self.keyword_check(lines)
        # extract the keyword appearances based on user preferences and save as max/mean/min
        max_keys, min_keys, mean_keys = self.extract_user_keys(users_db, keys)
        features_list = []
        # for each user append the articles one by one with the corresponding feature values
        for idx, _ in users_db.iterrows():
            for a in range(len(self.articles)):
                row = [
                    maximum_sim[idx][a],
                    minimum_sim[idx][a],
                    mean_sim[idx][a],
                    article_lengths[a],
                    max_keys[idx][a],
                    min_keys[idx][a],
                    mean_keys[idx][a]
                ]
                # use extend because otherwise we end up with a nested array which is not allowed in the
                # dimension reduction methods
                row.extend(check[a])
                # append the row to the overall feature list
                features_list.append(row)

        return features_list

    def reduce_dimension(self, features, labels, n_features, method='filter'):
        """
        Method to reduce the feature dimension to n features
        :param features: Array containing all features (one row per article)
        :param labels: Array with 1xN labels (like or dislike)
        :param n_features: number of features to keep (top n features will be selected. Only applies to filter or 
        wrapper methods
        :param method: Method to use for filtering. Choose between filter, wrapper or embedded. If nothing is set, filter
        methods will be used
        :return: The filtered features after dimension reduction
        """
        if method == 'filter':
            if os.path.isfile('./data/features_filtered{}.pickle'.format(n_features)):
                filtered_features = pickle.load(open("./data/features_filtered{}.pickle".format(n_features), "rb"))
            else:
                skb = SelectKBest(score_func=mutual_info_classif, k=n_features)
                skb.fit(features, labels)
                print('Feature scores according to mutual information:\n', skb.scores_)
                filtered_features = skb.transform(features)
                print("Before transformation: ", len(features),'x', len(features[0]), "After transformation: ",
                      filtered_features.shape)
                print('Compare: \n', features[0], '\n', filtered_features[0])
                pickle.dump(filtered_features, open("./data/features_filtered{}.pickle".format(n_features), "wb"))

        elif method == 'wrapper':
            if os.path.isfile('./data/features_wrapped{}.pickle'.format(n_features)):
                filtered_features = pickle.load(open("./data/features_wrapped{}.pickle".format(n_features), "rb"))
            else:
                model = sklearn.linear_model.LogisticRegression(random_state=42)
                rfe = RFE(model, n_features_to_select=n_features)
                rfe.fit(features, labels)
                idx1 = np.where(rfe.ranking_ == 1)[0][0]
                idx2 = np.where(rfe.ranking_ == 2)[0][0]
                idx3 = np.where(rfe.ranking_ == 3)[0][0]
                idx4 = np.where(rfe.ranking_ == 4)[0][0]
                idx5 = np.where(rfe.ranking_ == 5)[0][0]
                idx6 = np.where(rfe.ranking_ == 6)[0][0]
                idx7 = np.where(rfe.ranking_ == 7)[0][0]
                idx8 = np.where(rfe.ranking_ == 8)[0][0]
                idx9 = np.where(rfe.ranking_ == 9)[0][0]
                idx10 = np.where(rfe.ranking_ == 10)[0][0]
                print('Most promising features: ', idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10)
                filtered_features = rfe.transform(features)
                print("Before transformation: ", len(features), 'x', len(features[0]), "After transformation: ",
                      filtered_features.shape)
                print('Compare: \n', features[0], '\n', filtered_features[0])
                pickle.dump(filtered_features, open("./data/features_wrapped{}.pickle".format(n_features), "wb"))

        elif method == 'embedded':
            if os.path.isfile('./data/features_embedded.pickle'.format(n_features)):
                filtered_features = pickle.load(open("./data/features_embedded.pickle".format(n_features), "rb"))
            else:
                rf = RandomForestClassifier(n_estimators=10, random_state=42)
                rf.fit(features, labels)
                print('Feature importances of RF classifier: ', rf.feature_importances_)
                sfm = SelectFromModel(rf, threshold=0.1, prefit=True)
                filtered_features = sfm.transform(features)
                print("Before transformation: ", len(features),'x', len(features[0]), "After transformation: ",
                      filtered_features.shape)
                print('Compare: \n', features[3000], '\n', filtered_features[3000])
                pickle.dump(filtered_features, open("./data/features_embedded.pickle".format(n_features), "wb"))

        return filtered_features

    def train_classifier(self, features, kf, do_optimisation=False, nsplits=10):
        """
        Method to train the classifier
        :param features: The (filtered) features to be trained on
        :param kf: The k-fold split
        :param do_optimisation: set True if hyperparameter optimisation should be done. If not, best parameter will be used
        :param nsplits: integer indicating after how many splits during the hyperparameter \
        optimisation process the split should stop
        """
        # set up classifiers: use best parameter from previous optimisation(s) if do_optimisation is False

        if do_optimisation:
            # parameter grids for grid search optimisation  
            parameter_grid_knn = {'n_neighbors': np.arange(1, 21), 'weights': ['uniform', 'distance'], 'p': [1, 1.5, 2]}
            parameter_grid_rf = {'n_estimators': [8, 16, 32, 64, 128], 'criterion': ['gini', 'entropy']}

            classifiers = [
                ('kNN', KNeighborsClassifier()),
                ('RF', RandomForestClassifier())
            ]
        else:
            # for printing purposes
            parameter_grid_knn = "n_neighbors=8, weights='distance', p=1"
            parameter_grid_rf = "n_estimators=128, criterion='entropy'"

            classifiers = [
                ('kNN', KNeighborsClassifier(n_neighbors=8, weights='distance', p=1)),
                ('RF', RandomForestClassifier(n_estimators=128, criterion='entropy')) 
            ]



        split_counter = 1
        # training
        for name, model in classifiers:
            print("\nStart training with the {} classifier".format(name))
            print("Parameter:", parameter_grid_knn if name == 'kNN' else parameter_grid_rf)
            kappa_before = []
            precision_before = []
            recall_before = []
            
            # Used for comparison of hyperparameter optimisation
            best_parameter = []
            kappa_after = []
            precision_after = []
            recall_after = []
            
            confusion_matrix_result = []
            # split dataset n_splits times (see above)
            for train_index, test_index in kf.split(features):
                X_train = []; y_train = []
                X_test = []; y_test = []
                # for each split, get data
                for i in train_index:
                    X_train.append(features[i])
                    y_train.append(self.labels[i])
                for j in test_index:
                    X_test.append(features[j])
                    y_test.append(self.labels[j])

                # with data of split i, train the model and calculate the metrics
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                confusion_matrix_result.append([tn, fp, fn, tp])
                kappa_before.append(cohen_kappa_score(y_test, predictions))
                precision_before.append(precision_score(y_test, predictions))
                recall_before.append(recall_score(y_test, predictions))

                '''
                Hyperparameter Optimisation
                This takes really long for only one split. Therefore use nsplits for ability to stop earlier
                '''
                # depending on model, use different parameter grid
                if do_optimisation:  
                    print("\n  Before Grid-Search:", kappa_before[-1], precision_before[-1], recall_before[-1])
                    parameter_grid = parameter_grid_knn if name == 'kNN' else parameter_grid_rf
                    grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring=make_scorer(cohen_kappa_score))
                    grid_search.fit(X_train, y_train)
                    print('  Best parameters:', grid_search.best_params_)
                    predictions = grid_search.predict(X_test)
                    kappa_after = (cohen_kappa_score(y_test, predictions))
                    precision_after = (precision_score(y_test, predictions))
                    recall_after = (recall_score(y_test, predictions))
                    print("  After Grid-Search:", name, kappa_after, precision_after, recall_after)

                    if split_counter == nsplits:
                        split_counter = 0
                        break
                    split_counter += 1
            # take the mean over all splits
            if not do_optimisation:
                print("  Results:\n        Cohen's Kappa         Precision          Recall\n", \
                    name, np.mean(kappa_before), np.mean(precision_before), np.mean(recall_before))
            # used to check performances when inserting optimised parameters for each split
            print("\n  Confusion Matrix:\n  TN     FP    FN     TP")
            for m in confusion_matrix_result:
                print(m)

