import pandas as pd
import matplotlib.pyplot as plt


def calculate(model):
    data = pd.read_csv(model + '.csv')
    data['Accuracy'] = data['correctPredictions'] / (data['totalPredictions'] + data['totalAnswers'] - data['correctPredictions'])
    data['Precision'] = data['correctPredictions'] / data['totalAnswers']
    data['Recall'] = data['correctPredictions'] / data['totalPredictions']
    data['F1'] = 2 * data['correctPredictions'] / (data['totalPredictions'] + data['totalAnswers'])
    return data


def show_plot(statistic, dfs, all_models):
    plt.figure()
    plt.ylim(0, 1)
    plt.title(statistic)
    values = []
    for mod in all_models:
        val = dfs[mod][statistic].mean()
        values.append(val)
        plt.text(mod, val + 0.02, round(val, 3), ha='center')
    plt.bar(models, values)
    plt.show()


dataframes = {}
models = ['CNN14', 'ResNet22', 'DaiNet19', 'MobileNetV1']
for model in models:
    dataframes[model] = calculate(model)

show_plot('Accuracy', dataframes, models)
show_plot('Precision', dataframes, models)
show_plot('Recall', dataframes, models)
show_plot('F1', dataframes, models)

plt.figure()
plt.title('Accuracy compared to time taken to process 1000 wav files')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.xlabel('Minutes')
values = []
time = []
for model in models:
    model_time = dataframes[model]['prediction_time_seconds'].sum() / 60
    model_acc = dataframes[model]['Accuracy'].mean()
    time.append(model_time)
    values.append(model_acc)
    plt.text(model_time, model_acc + 0.01, model)
plt.plot(time, values, 'ro')
plt.show()
