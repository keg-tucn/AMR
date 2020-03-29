import matplotlib.pyplot as plt
import numpy as np

from definitions import RESULT_PLOTS_DIR


def plot_history(history, model_name, trial_name):
    _plot_accuracy(history, model_name, trial_name)

    _plot_loss(history, model_name, trial_name)


def _plot_accuracy(history, model_name, trial_name):
    _config_plot(history.params.get('epochs'))
    plt.title('{} accuracy'.format(model_name))
    plt.ylabel('accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(RESULT_PLOTS_DIR + '/accuracy_{}_{}.png'.format(model_name, trial_name))


def _plot_loss(history, model_name, trial_name):
    _config_plot(history.params.get('epochs'))
    plt.title('{} loss'.format(model_name))
    plt.ylabel('loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(RESULT_PLOTS_DIR + '/loss_{}_{}.png'.format(model_name, trial_name))


def _config_plot(x_lim):
    plt.clf()
    plt.autoscale(False)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, x_lim)
    plt.ylim(0.0, 1.0)
    plt.xlabel('epoch')

    ax = plt.gca()

    major_x_ticks = np.arange(0, x_lim + 1, 5)
    minor_x_ticks = np.arange(0, x_lim + 1, 1)
    ax.set_xticks(major_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)

    major_y_ticks = np.arange(0, 1.025, 0.1)
    minor_y_ticks = np.arange(0, 1.025, 0.025)
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)

    ax.grid(which='major')
    ax.grid(which='minor', alpha=0.3)
