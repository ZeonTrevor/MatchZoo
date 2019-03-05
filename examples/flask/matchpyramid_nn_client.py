import requests
import numpy as np
import drmm_nn_client

class NN_client:

    def __init__(self, url):
        self.url = url
        self.headers = {'Accept': 'application/json', 'content-type': 'application/json'}

    def score_doc(self, q, d):
        data = dict()
        data["query"] = q
        data["doc"] = d
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "score", json=data, headers=self.headers)
        return float(response.text)

    def score_doc_vec(self, q, d):
        data = dict()
        data["query"] = q
        data["doc"] = d
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "score_doc_vec", json=data, headers=self.headers)
        return float(response.text)

    def transform_doc(self, d):
        data = dict()
        data["doc"] = d
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "transform_doc", json=data, headers=self.headers).json()
        return response["doc_vec"]

    def transform_doc_vec(self, doc_vec):
        data = dict()
        data["doc_vec"] = doc_vec
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "transform_doc_vec", json=data, headers=self.headers).json()
        return response["doc"]

    # def fetch_background_data(self):
    #     self.headers['Connection'] = 'close'
    #     response = requests.get(self.url + "fetch_data", headers=self.headers).json()
    #     background_data = response["background_data"]
    #     print(type(background_data))
    #     print(len(background_data))
    #     print(np.asarray(background_data).shape)
    #     # print(background_data[0])
    #     return np.asarray(background_data)

    def fetch_model_input_data(self):
        self.headers['Connection'] = 'close'
        response = requests.get(self.url + "fetch_model_input_data", headers=self.headers).json()
        query = response["query"]
        doc = response["doc"]
        # dpool_index = response["dpool_index"]
        print(np.asarray(query).shape)
        print(np.asarray(doc).shape)
        # print(np.asarray(dpool_index).shape)

        return np.asarray(query), np.asarray(doc)  # , np.asarray(dpool_index)

    def prepare_test_input_data(self, query, doc):
        data = dict()
        data["query"] = query
        data["doc"] = doc
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "prepare_test_input", json=data, headers=self.headers).json()
        query = response["query"]
        doc = response["doc"]
        # dpool_index = response["dpool_index"]
        # print(np.asarray(query).shape)
        # print(np.asarray(doc).shape)
        # print(np.asarray(dpool_index).shape)
        return np.asarray(query), np.asarray(doc)  # , np.asarray(dpool_index)


if __name__ == '__main__':
    cli = NN_client("http://127.0.0.1:5008/")
    drmm_cli = drmm_nn_client.NN_client("http://127.0.0.1:5007/")
    # print(cli.score_doc("meeting", "hi greetings nice to meet you"))

    # cli.fetch_background_data()
    Q, D = cli.fetch_model_input_data()

    doc_cont = drmm_cli.get_doc_content("FBIS3-10082")
    # q, d = cli.prepare_test_input_data("international organized crime", doc_cont)

    # doc_vec = cli.transform_doc(doc_cont)
    # print(len(doc_vec))
    # print(doc_vec.count(106662))
    # print(doc_vec)
    # print(doc_cont)
    # print(cli.transform_doc_vec(doc_vec))
    # print(len(doc_cont.split()))
    print(cli.score_doc("international organized crime", doc_cont))
    # print(cli.score_doc_vec("international organized crime", doc_vec))
