import os, inspect

import tensorflow as tf
import numpy as np

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(agent, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    iteration = 0

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttv=0)

            if(minibatch['x'].shape[0] == 0): break
            loss, accuracy, class_score = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            iteration += 1

            if(minibatch['terminate']): break

        agent.save_params(model='final_epoch')

        print("Epoch [%d / %d] (%d iteration)  Loss:%.5f, Acc:%.5f" \
            %(epoch, epochs, iteration, loss, accuracy))
    agent.save_params(model='final_epoch')

def test(agent, dataset):

    list_model = ['base', 'final_epoch']

    best_f1 = 0
    for idx_model, name_model in enumerate(list_model):
        print(name_model)
        try: agent.load_params(model=name_model)
        except: print("Parameter loading was failed")
        else: print("Parameter loaded")

        print("\nTest...")

        confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
        while(True):

            minibatch = dataset.next_batch(batch_size=1, ttv=1)
            if(minibatch['x'].shape[0] == 0): break
            loss, accuracy, class_score = agent.step(minibatch=minibatch, training=False)
            y_te = minibatch['y']

            label, logit = np.argmax(y_te[0]), np.argmax(class_score)
            confusion_matrix[label, logit] += 1

            if(minibatch['terminate']): break

        print("\nConfusion Matrix")
        print(confusion_matrix)

        tot_precision, tot_recall, tot_f1score = 0, 0, 0
        diagonal = 0
        for idx_c in range(dataset.num_class):
            precision = confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[:, idx_c])
            recall = confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[idx_c, :])
            f1socre = 2 * (precision * recall / (precision + recall))

            tot_precision += precision
            tot_recall += recall
            tot_f1score += f1socre
            diagonal += confusion_matrix[idx_c, idx_c]
            print("Class-%d | Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
                %(idx_c, precision, recall, f1socre))

        accuracy = diagonal / np.sum(confusion_matrix)
        print("\nTotal | Accuracy: %.5f, Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
            %(accuracy, tot_precision/dataset.num_class, tot_recall/dataset.num_class, tot_f1score/dataset.num_class))

        best_f1 = max(best_f1, tot_f1score)

    return best_f1, len(list_model)
