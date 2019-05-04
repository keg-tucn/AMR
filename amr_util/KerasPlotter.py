import matplotlib.pyplot as plt

from definitions import RESULT_PLOTS_DIR


def plot_history(history, model_name, trial_name):
    plt.ylim(0.0, 1.0)
    plt.clf()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('{} accuracy'.format(model_name))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(RESULT_PLOTS_DIR + '/accuracy_{}_{}.png'.format(model_name, trial_name))
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('{} loss'.format(model_name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(RESULT_PLOTS_DIR + '/loss_{}_{}.png'.format(model_name, trial_name))
