from dictionary import Dictionary
from ldamodel import LDAModel
import math
import matplotlib.pyplot as plt
import copy


class EvalTools:

    def __init__(self, dic_doc: Dictionary, save_point, epss, line_names):
        self.dic_doc = dic_doc
        self.model_series = []
        self.save_point = save_point
        self.epss = epss
        self.end_point = save_point[-1]
        self.line_names = line_names

    def add_series(self, model_name):
        one = []
        for each in self.save_point:
            model = LDAModel().load_model(model_name, each)
            one.append(model)
        self.model_series.append(one)

    def add_eps_perp(self, model_names):
        one = []
        for model in model_names:
            model = LDAModel().load_model(model, self.end_point)
            one.append(model)
        self.model_series.append(one)

    def show_topic2word(self, model_name, n_iter, topic):
        model = LDAModel().load_model(model_name, n_iter)
        nkv = model.get_nkv()
        K = len(nkv)
        if topic >= K:
            return
        V = len(nkv[0])
        y = copy.copy(nkv[topic])
        y.sort(reverse=True)
        yp = []
        for i in range(V):
            if y[i] > 0:
                yp.append(y[i])
            else:
                break
        plt.plot([v for v in range(len(yp))], yp)
        plt.show()

    def draw(self, x, title="", x_name="", y_name="", file_name="test.pdf"):
        markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
        mecs = ['red', 'blue', 'coral', 'darkgreen', 'firebrick', 'orange', 'indigo', 'lavender', 'darkviolet',
                'indianred', 'khaki']
        colors = ['red', 'blue', 'coral', 'darkgreen', 'firebrick', 'orange', 'indigo', 'lavender', 'darkviolet',
                  'indianred', 'khaki']
        for i in range(len(self.model_series)):
            name = self.line_names[i]
            print("drawing series of", name, "...")
            # 循环一次画一条线
            y = [self.model_series[i][j].get_perplexity(self.save_point[j]) for j in range(len(self.model_series[i]))]
            print(y)
            plt.plot(x, y, marker=markers[i], mec=mecs[i], mfc='w', color=colors[i], label=name)
        plt.legend()
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(x_name)
        plt.ylabel(y_name)  # Y轴标签
        plt.title(title)  # 标题
        fig1 = plt.gcf()
        plt.show()
        plt.close()
        fig1.savefig(file_name, dpi=100)

