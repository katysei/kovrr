import json
import requests, bs4
import logging
from string import Template
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_numeric, strip_punctuation, \
    strip_short
from sklearn.model_selection import TimeSeriesSplit
import sys

HN_URL_TOP_STORIES_IDS = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_URL_STORY = "https://hacker-news.firebaseio.com/v0/item/$id.json"
HN_NUM_STORIES = 40
TECHNOLOGIES = ["Kubernetes", "Linux", "Windows", "Solarwinds", "Garmin", "AWS", "Docker", "Github", "Wordpress",
                "Rundeck"]
HN_FILE_NAME = 'hacker_news_data.json'
HN_COLUMNS = ["time", "descendants"]

logging.basicConfig(level=logging.ERROR)


def get_json(url):
    res = requests.get(url).json()
    return res


def get_ids(url):
    return get_json(url)


def gen_item_url(url_id, id):
    """
    :param url_id:
    :param id:
    :return: generating API url for extracting story data
    """
    template_url = Template(url_id)
    url = template_url.substitute(id=id)
    return url


def get_item(url_id, id):
    """
    :param url_id:
    :param id:
    :return:story using story id
    """
    url = gen_item_url(url_id, id)
    item = get_json(url)
    return item


def get_HN_best_stories(url_stories_ids, url_id, n=40):
    get_stories(url_stories_ids, url_id, n)


def get_stories(url_stories_ids, url_id, n=40):
    """
    :param url_stories_ids:
    :param url_id:
    :param n:
    :return: prints top stories using HN API
    """
    ids = get_json(url_stories_ids)[:n]
    for i, id in enumerate(ids):
        story = get_item(url_id, id)
        title = story["title"]
        rank = i + 1
        print(f"Rank:{rank} Title:{title}")


def get_datetime(unixtime):
    dt = datetime.datetime.fromtimestamp(unixtime)
    return dt


def get_hours_diff(unixtime):
    """

    :param unixtime:
    :return: diff between input time  and 20PM in the same day
    """
    dt = datetime.datetime.fromtimestamp(unixtime)
    tm = datetime.time(20, 00)
    dt_20pm = datetime.datetime.combine(dt.date(), tm)
    return (dt - dt_20pm).total_seconds() / 3600


def get_data(filename, keys):
    """
    reads a HN datafile and
    :param filename:
    :param keys:
    :return: list records
    """
    data = []
    json_data = None
    try:
        with open(filename) as json_file:
            json_data = json.load(json_file)
    except Exception as e:
        logging.error("Cannot read file" + str(e))
        return None
    errors = 0
    n = 0
    for item in json_data:
        n += 1
        rec = []
        try:
            for k in keys:
                rec = rec + [item[k]]
        except Exception as e:
            errors += 1
            # logging.error('Key Error:'+ str(e))
            continue
        data.append(rec)
    logging.info(f"No of errors {errors},Percentage {(errors / n):.2f}")
    return data


def get_corr(filename, keys):
    """

    :param filename:
    :param keys:
    :return: correaltion between features extractd from the file
    """
    data = get_data(filename, keys=HN_COLUMNS)
    df = pd.DataFrame(data=data, columns=keys)
    df = df.rename(columns={'time': 'timestamp', 'descendants': 'comments'})
    df["dt"] = df["timestamp"].apply(lambda x: get_datetime(x))
    df["month"] = df["dt"].apply(lambda x: x.month)
    df["year"] = df["dt"].apply(lambda x: x.year)
    df["diff_8pm"] = df["timestamp"].apply(lambda x: abs(get_hours_diff(x)))
    corr, _ = pearsonr(df["diff_8pm"], df["comments"])
    print(f"Correlation between hours difference from 20PM and # of comments {corr:.6f}")
    plot_corr_xy(df)


def plot_corr_xy(df):
    _ = sns.scatterplot(data=df, y="comments", x="diff_8pm")


def preprocess(df):
    """
    :param df:
    :return: preprocessed df
    """
    df["title"] = df["title"].apply(lambda x: remove_stopwords(x))
    df["title"] = df["title"].apply(lambda x: strip_numeric(x))
    df["title"] = df["title"].apply(lambda x: strip_punctuation(x))
    df["title"] = df["title"].apply(lambda x: x.lower())
    df["title"] = df["title"].apply(lambda x: strip_short(x, minsize=3))
    df["dt"] = df["timestamp"].apply(lambda x: get_datetime(x))
    df["month"] = df["dt"].apply(lambda x: x.month)
    df["year"] = df["dt"].apply(lambda x: x.year)
    return df


def get_features(df):
    """
    generating dataframes with features to be input for a model
    :param df:
    :return: dataframe
    """
    corpus = df["title"]
    vectorizer = CountVectorizer(min_df=4, max_df=0.1, max_features=1000, binary=True)
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    df_f = pd.DataFrame(data=X.toarray(), columns=features)
    df_f["month"] = df["month"]
    df_f["year"] = df["year"]
    df_f["count"] = 1
    dg = df_f.groupby(["year", "month"]).mean()
    dg = dg.reset_index()
    return dg


def gen_HN_dataset(filename):
    """
    generating HN dataset from json file
    :param filename:
    :return: dataframe
    """
    data = get_data(filename, keys=["time", "title"])
    df = pd.DataFrame(data=data, columns=["time", "title"])
    df = df.rename(columns={'time': 'timestamp'})
    df = preprocess(df)
    df = get_features(df)
    return df


def generate_tech_str():
    """
    :return: str prompt for technology input
    """
    s = []
    for i, t in enumerate(TECHNOLOGIES):
        s.append("  " + str(i) + "." + t)
    return "\n".join(s)


def get_input():
    """
    get command line input for technology to be predicted
    :return: index in TECHNOLOGIES list pointing to the technology
    """
    while True:
        tech_str = generate_tech_str()
        print(tech_str)
        n = input("Choose a technology:")
        try:
            n = int(n)
        except ValueError:
            print("Error: Not an integer...\n5")
            continue
        if n not in range(0, len(TECHNOLOGIES)):
            print(f"Error: Not in range [{0},{len(TECHNOLOGIES) - 1}]")
            continue
        return n


def time_train_test_split(X, y):
    """
    :param X: input data
    :param y: output data
    :return: time series data splited into train and test
    """
    n = X.shape[0]
    n_train = int(n * 0.6)
    xr_train = np.array(range(0, n_train))
    yr_train = xr_train + 1
    n_test = n_train
    xr_test = np.array(range(n_test, n - 1))
    yr_test = xr_test + 1
    X_train = X.iloc[xr_train, :]
    y_train = y[yr_train]
    X_test = X.iloc[xr_test, :]
    y_test = y[yr_test]
    return X_train, X_test, y_train, y_test


""" 
def time_train_test_split_2(X):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    tscv = TimeSeriesSplit(n_splits=2)
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
"""


def predict_tech_next_month(filename):
    """
    generate dataset from file and predicts next month probability
    based upon input feature
    :param filename:
    :return: model
    """
    n = get_input()
    df = gen_HN_dataset(filename)
    y_feature = TECHNOLOGIES[n].lower()
    if y_feature not in df.columns:
        logging.error(f"Not enough data on {TECHNOLOGIES[n]}")
        return (-1)
    X = df
    y = df[y_feature]
    X_train, X_test, y_train, y_test = time_train_test_split(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    print(f"{TECHNOLOGIES[n]} probability true: {y_test.values}  probability prediction :{y_test_predict}")
    y_pred = model.predict(X)
    y_true = y
    _, r2 = pearsonr(y_true, y_pred)
    print(f"corr {r2}")
    print(f"{TECHNOLOGIES[n]} next month probability prediction :{model.predict(X.iloc[-1:, :])}")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("# of Argumnets is not 1, use top for top stories  corr for correlation nextmonth for prediction.")
        exit(-1)
    if sys.argv[1] not in ["top", "corr"]:
        print("Use top for top stories  corr for correlation.")
        exit(-1)
    if sys.argv[1] == "top":
        get_HN_best_stories(HN_URL_TOP_STORIES_IDS, HN_URL_STORY, HN_NUM_STORIES)
        exit(0)
    if sys.argv[1] == "corr":
        get_corr(HN_FILE_NAME, HN_COLUMNS)
        exit(0)
    if sys.argv[1] == "nextmonth":
        predict_tech_next_month(HN_FILE_NAME)
        exit(0)

    exit(-1)
