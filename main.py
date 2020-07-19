import pandas as pd

from models import Models
from ablation_analysis import Ablation

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
del train_data['word']
del test_data['word']

X_train = train_data.iloc[:, :-1]
y_train = train_data.label
X_test = test_data.iloc[:, :-1]
y_test = test_data.label


for num in range(1, 6):
    model = Models.get_model(num)
    final_metrics = dict()

    ablation = Ablation(X_train, y_train, X_test, y_test, model)
    final_metrics[Models.get_model_name(num)] = ablation.run_test()

    with open('analysis' + str(num) + '.csv', 'w') as file:
        for model, features in final_metrics.items():
            for feature, metrics in features.items():
                for m_name, m_value in metrics.items():
                    file.write(model + "," + feature + "," + m_name + "," + str(m_value) + "\n")

