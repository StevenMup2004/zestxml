import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tabulate import tabulate
from io import StringIO
from tqdm import tqdm
from utils import *
import json


def _filter(score_mat, filter_mat, copy=True):
    if filter_mat is None:
        return score_mat
    if copy:
        score_mat = score_mat.copy()

    temp = filter_mat.tocoo()
    score_mat[temp.row, temp.col] = 0
    del temp
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat

dataset = sys.argv[1]
dataset_path = sys.argv[2]
RES_DIR = f'{dataset_path}/Results/{dataset}'
# DATA_DIR = f'GZXML-Datasets/{dataset}'
DATA_DIR = f'{dataset_path}'

# print(_c("Loading files", attr="yellow"))
print("Loading files")
trn_X_Y = read_sparse_mat('%s/trn_X_Y.txt'%DATA_DIR, use_xclib=False)
tst_X_Y = read_sparse_mat('%s/tst_X_Y.txt'%DATA_DIR, use_xclib=False)

score_mat = _filter(read_bin_spmat(f'{RES_DIR}/score_mat.bin').copy(), None)
# Shape should be:
# nrows = number of test data
# ncols = scores for possible labels
x = score_mat.toarray()

# getting the set of seen labels in training dataset
seen_labels = set()
train_label = []
with open(f'{DATA_DIR}/trn_X_Y.txt', "r", encoding="utf-8") as re:
    train_label = re.readlines()[1:]
for text in train_label:
    list_labels = []
    split = text.split(" ")
    for label in split:
        label_num = label.split(":")[0]
        seen_labels.add(int(label_num))


# loop through the score matrix
text_labels = []
with open(f'{DATA_DIR}/tst_X_Y.txt', "r", encoding="utf-8") as re:
    text_labels = re.readlines()[1:]
actuals = []
iter2= 0

f_unrep = open("fix_new_id_repo_unrepresentative_5_test.txt")
list_of_unrep = f_unrep.read().splitlines()
fin_dict = {}
for line in list_of_unrep:
    temp_split_arr = line.split("\t")
    fin_dict[int(temp_split_arr[0])] = int(temp_split_arr[2])
f_unrep.close()
num_of_wrong = 0
print(fin_dict)
for text in text_labels:
    list_labels = []
    split = text.split(" ")
    for label in split:
        label_num = label.split(":")[0]
        list_labels.append(int(label_num))
    actuals.append(list_labels)
    #print(iter2)
    # if len(list_labels) == 19:
    #     print(iter2)
    if iter2 in fin_dict:
        if fin_dict[iter2] != len(list_labels):
            num_of_wrong += 1
    iter2 += 1
print(num_of_wrong)
sum_sr_1 = 0
sum_sr_2 = 0
sum_sr_3 = 0
sum_sr_4 = 0
sum_sr_5 = 0
sum_sr_6 = 0
sum_sr_7 = 0
sum_sr_8 = 0
sum_sr_9 = 0
sum_sr_10 = 0
sum_recall_1 = 0
sum_recall_2 = 0
sum_recall_3 = 0
sum_recall_4 = 0
sum_recall_5 = 0
sum_recall_6 = 0
sum_recall_7 = 0
sum_recall_8 = 0
sum_recall_9 = 0
sum_recall_10 = 0
sum_precision_1 = 0
sum_precision_2 = 0
sum_precision_3 = 0
sum_precision_4 = 0
sum_precision_5 = 0
sum_precision_6 = 0
sum_precision_7 = 0
sum_precision_8 = 0
sum_precision_9 = 0
sum_precision_10 = 0
num_test_data = 0
total_labels = 0

prediction_not_seen = {}
prediction_not_seen_correct = {}

threshold = [5]#,10,20,30,40,50,60,70,80,90,100]
for thres in threshold:
    f_unrep = open("fix_new_id_repo_unrepresentative_"+str(thres)+"_test.txt")
    list_of_unrep = f_unrep.read().splitlines()
    fin_dict = {}
    for line in list_of_unrep:
        temp_split_arr = line.split("\t")
        fin_dict[int(temp_split_arr[0])] = int(temp_split_arr[2])
    f_unrep.close()
    # list_of_unrep = f_unrep.read().splitlines()
    # list_unrep = list(map(int, list_of_unrep))
    # f_unrep.close()

    f_results = open("ZestXML_compare_"+str(thres)+".txt", "w")
    for i, rows in enumerate(x):
        if i in fin_dict:
            predictions = np.argpartition(rows, -10)[-10:]
            predictions = predictions[::-1]
            # print(predictions)
            temp_pred = rows[predictions[:10]]
            new_sort = np.argsort(temp_pred)
            ns = list(new_sort)
            # print(new_sort)

            num_test_data += 1
            local_correct_prediction = 0
            labels = actuals[i]
            # get only the top k prediction

            total_labels += len(labels)
            correct_prediction = 0

            # print("Prediction labels")
            #print("SCORES")
            #print(temp_pred)
            #print("BEFORE")
            #print(predictions)
            #print("AFTER")
            #print(new_sort)
            # print([predictions[ns[len(ns)-1]],predictions[ns[len(ns)-2]],predictions[ns[len(ns)-3]]])
            # print(labels)
            # print()
            # print([predictions[ns[len(ns) - 1]], predictions[ns[len(ns) - 2]], predictions[ns[len(ns) - 3]]])
            # print(labels)
            # print()
            temp_sr_1 = 0
            f_results.write(str(i) + "\t")
            if predictions[ns[len(ns)-1]] in labels:
                temp_sr_1 += 1
                sum_sr_1 += 1
                sum_sr_2 += 1
                sum_sr_3 += 1
                sum_sr_4 += 1
                sum_sr_5 += 1
                sum_sr_6 += 1
                sum_sr_7 += 1
                sum_sr_8 += 1
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-2]] in labels:
                sum_sr_2 += 1
                sum_sr_3 += 1
                sum_sr_4 += 1
                sum_sr_5 += 1
                sum_sr_6 += 1
                sum_sr_7 += 1
                sum_sr_8 += 1
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-3]] in labels:
                sum_sr_3 += 1
                sum_sr_4 += 1
                sum_sr_5 += 1
                sum_sr_6 += 1
                sum_sr_7 += 1
                sum_sr_8 += 1
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-4]] in labels:
                sum_sr_4 += 1
                sum_sr_5 += 1
                sum_sr_6 += 1
                sum_sr_7 += 1
                sum_sr_8 += 1
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-5]] in labels:
                sum_sr_5 += 1
                sum_sr_6 += 1
                sum_sr_7 += 1
                sum_sr_8 += 1
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-6]] in labels:
                sum_sr_6 += 1
                sum_sr_7 += 1
                sum_sr_8 += 1
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-7]] in labels:
                sum_sr_7 += 1
                sum_sr_8 += 1
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-8]] in labels:
                sum_sr_8 += 1
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-9]] in labels:
                sum_sr_9 += 1
                sum_sr_10 += 1
            elif predictions[ns[len(ns)-10]] in labels:
                sum_sr_10 += 1

            if temp_sr_1 > 0:
                f_results.write("1\n")
            else:
                f_results.write("0\n")



            # K = 1
            if predictions[ns[len(ns)-1]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-1]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-1]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-1]], 0) + 1
            if predictions[ns[len(ns)-1]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-1]]] = prediction_not_seen.get(predictions[ns[len(ns)-1]], 0) + 1
            # sum_precision_1 += (correct_prediction / min(1, len(labels)))
            sum_precision_1 += (correct_prediction / 1)
            sum_recall_1 += (correct_prediction / len(labels))

            # K = 2
            if predictions[ns[len(ns)-2]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-2]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-2]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-2]], 0) + 1
            if predictions[ns[len(ns)-2]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-2]]] = prediction_not_seen.get(predictions[ns[len(ns)-2]], 0) + 1
            # sum_precision_2 += (correct_prediction / (min(2, len(labels))))
            sum_precision_2 += (correct_prediction / 2)
            sum_recall_2 += (correct_prediction / len(labels))

            # K = 3
            if predictions[ns[len(ns)-3]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-3]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-3]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-3]], 0) + 1
            if predictions[ns[len(ns)-3]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-3]]] = prediction_not_seen.get(predictions[ns[len(ns)-3]], 0) + 1
            # sum_precision_3 += (correct_prediction / min(3, len(labels)))
            sum_precision_3 += (correct_prediction / 3)
            sum_recall_3 += (correct_prediction / len(labels))

            # K = 4
            if predictions[ns[len(ns)-4]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-4]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-4]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-4]], 0) + 1
            if predictions[ns[len(ns)-4]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-4]]] = prediction_not_seen.get(predictions[ns[len(ns)-4]], 0) + 1
            # sum_precision_4 += (correct_prediction / min(4, len(labels)))
            sum_precision_4 += (correct_prediction / 4)
            sum_recall_4 += (correct_prediction / len(labels))

            # K = 5
            if predictions[ns[len(ns)-5]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-5]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-5]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-5]], 0) + 1
            if predictions[ns[len(ns)-5]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-5]]] = prediction_not_seen.get(predictions[ns[len(ns)-5]], 0) + 1
            # sum_precision_5 += (correct_prediction / min(5, len(labels)))
            sum_precision_5 += (correct_prediction / 5)
            sum_recall_5 += (correct_prediction / len(labels))


            # K = 6
            if predictions[ns[len(ns)-6]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-6]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-6]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-6]], 0) + 1
            if predictions[ns[len(ns)-6]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-6]]] = prediction_not_seen.get(predictions[ns[len(ns)-6]], 0) + 1
            # sum_precision_6 += (correct_prediction / min(6, len(labels)))
            sum_precision_6 += (correct_prediction / 6)
            sum_recall_6 += (correct_prediction / len(labels))


            # K = 7
            if predictions[ns[len(ns)-7]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-7]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-7]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-7]], 0) + 1
            if predictions[ns[len(ns)-7]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-7]]] = prediction_not_seen.get(predictions[ns[len(ns)-7]], 0) + 1
            # sum_precision_7 += (correct_prediction / min(7, len(labels)))
            sum_precision_7 += (correct_prediction / 7)
            sum_recall_7 += (correct_prediction / len(labels))


            # K = 8
            if predictions[ns[len(ns)-8]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-8]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-8]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-8]], 0) + 1
            if predictions[ns[len(ns)-8]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-8]]] = prediction_not_seen.get(predictions[ns[len(ns)-8]], 0) + 1
            # sum_precision_8 += (correct_prediction / min(8, len(labels)))
            sum_precision_8 += (correct_prediction / 8)
            sum_recall_8 += (correct_prediction / len(labels))


            # K = 9
            if predictions[ns[len(ns)-9]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-9]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-9]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-9]], 0) + 1
            if predictions[ns[len(ns)-9]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-9]]] = prediction_not_seen.get(predictions[ns[len(ns)-9]], 0) + 1
            # sum_precision_9 += (correct_prediction / min(9, len(labels)))
            sum_precision_9 += (correct_prediction / 9)
            sum_recall_9 += (correct_prediction / len(labels))


            # K = 10
            if predictions[ns[len(ns)-10]] in labels:
                correct_prediction += 1
                if predictions[ns[len(ns)-10]] not in seen_labels:
                    prediction_not_seen_correct[predictions[ns[len(ns)-10]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-10]], 0) + 1
            if predictions[ns[len(ns)-10]] not in seen_labels:
                prediction_not_seen[predictions[ns[len(ns)-10]]] = prediction_not_seen.get(predictions[ns[len(ns)-10]], 0) + 1
            # sum_precision_10 += (correct_prediction / min(10, len(labels)))
            sum_precision_10 += (correct_prediction / 10)
            sum_recall_10 += (correct_prediction / len(labels))
    precision_1 = sum_precision_1 / num_test_data
    precision_2 = sum_precision_2 / num_test_data
    precision_3 = sum_precision_3 / num_test_data
    precision_4 = sum_precision_4 / num_test_data
    precision_5 = sum_precision_5 / num_test_data
    recall_1 = sum_recall_1 / num_test_data
    recall_2 = sum_recall_2 / num_test_data
    recall_3 = sum_recall_3 / num_test_data
    recall_4 = sum_recall_4 / num_test_data
    recall_5 = sum_recall_5 / num_test_data
    sr_1 = sum_sr_1 / num_test_data
    sr_2 = sum_sr_2 / num_test_data
    sr_3 = sum_sr_3 / num_test_data
    sr_4 = sum_sr_4 / num_test_data
    sr_5 = sum_sr_5 / num_test_data
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)
    f1_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3)
    f1_4 = 2 * precision_4 * recall_4 / (precision_4 + recall_4)
    f1_5 = 2 * precision_5 * recall_5 / (precision_5 + recall_5)


    precision_6 = sum_precision_6 / num_test_data
    recall_6 = sum_recall_6 / num_test_data
    sr_6 = sum_sr_6 / num_test_data
    f1_6 = 2 * precision_6 * recall_6 / (precision_6 + recall_6)


    precision_7 = sum_precision_7 / num_test_data
    recall_7 = sum_recall_7 / num_test_data
    sr_7 = sum_sr_7 / num_test_data
    f1_7 = 2 * precision_7 * recall_7 / (precision_7 + recall_7)


    precision_8 = sum_precision_8 / num_test_data
    recall_8 = sum_recall_8 / num_test_data
    sr_8 = sum_sr_8 / num_test_data
    f1_8 = 2 * precision_8 * recall_8 / (precision_8 + recall_8)


    precision_9 = sum_precision_9 / num_test_data
    recall_9 = sum_recall_9 / num_test_data
    sr_9 = sum_sr_9 / num_test_data
    f1_9 = 2 * precision_9 * recall_9 / (precision_9 + recall_9)


    precision_10 = sum_precision_10 / num_test_data
    recall_10 = sum_recall_10 / num_test_data
    sr_10 = sum_sr_10 / num_test_data
    f1_10 = 2 * precision_10 * recall_10 / (precision_10 + recall_10)

    f_write_results = open("zestxml_combined_threshold_"+str(thres)+".txt", "w")
    f_write_results.write("K = 1\n")
    f_write_results.write("P@1 = " + precision_1.__str__()+"\n")
    f_write_results.write("R@1 = " + recall_1.__str__()+"\n")
    f_write_results.write("F@1 = " + f1_1.__str__()+"\n")
    f_write_results.write("SR@1 = " + sr_1.__str__()+"\n")

    f_write_results.write("K = 2\n")
    f_write_results.write("P@2 = " + precision_2.__str__()+"\n")
    f_write_results.write("R@2 = " + recall_2.__str__()+"\n")
    f_write_results.write("F@2 = " + f1_2.__str__()+"\n")
    f_write_results.write("SR@2 = " + sr_2.__str__()+"\n")

    f_write_results.write("K = 3\n")
    f_write_results.write("P@3 = " + precision_3.__str__()+"\n")
    f_write_results.write("R@3 = " + recall_3.__str__()+"\n")
    f_write_results.write("F@3 = " + f1_3.__str__()+"\n")
    f_write_results.write("SR@3 = " + sr_3.__str__()+"\n")
    f_write_results.write("TOTAL LABELS: " + total_labels.__str__()+"\n")

    f_write_results.write("K = 4\n")
    f_write_results.write("P@4 = " + precision_4.__str__()+"\n")
    f_write_results.write("R@4 = " + recall_4.__str__()+"\n")
    f_write_results.write("F@4 = " + f1_4.__str__()+"\n")
    f_write_results.write("SR@4 = " + sr_4.__str__()+"\n")
    f_write_results.write("TOTAL DATA: " + num_test_data.__str__()+"\n")

    f_write_results.write("K = 5\n")
    f_write_results.write("P@5 = " + precision_5.__str__()+"\n")
    f_write_results.write("R@5 = " + recall_5.__str__()+"\n")
    f_write_results.write("F@5 = " + f1_5.__str__()+"\n")
    f_write_results.write("SR@5 = " + sr_5.__str__()+"\n")
    f_write_results.write("TOTAL LABELS: " + total_labels.__str__()+"\n")

    f_write_results.write("K = 6\n")
    f_write_results.write("P@6 = " + precision_6.__str__()+"\n")
    f_write_results.write("R@6 = " + recall_6.__str__()+"\n")
    f_write_results.write("F@6 = " + f1_6.__str__()+"\n")
    f_write_results.write("SR@6 = " + sr_6.__str__()+"\n")

    f_write_results.write("K = 7\n")
    f_write_results.write("P@7 = " + precision_7.__str__()+"\n")
    f_write_results.write("R@7 = " + recall_7.__str__()+"\n")
    f_write_results.write("F@7 = " + f1_7.__str__()+"\n")
    f_write_results.write("SR@7 = " + sr_7.__str__()+"\n")

    f_write_results.write("K = 8\n")
    f_write_results.write("P@8 = " + precision_8.__str__()+"\n")
    f_write_results.write("R@8 = " + recall_8.__str__()+"\n")
    f_write_results.write("F@8 = " + f1_8.__str__()+"\n")
    f_write_results.write("SR@8 = " + sr_8.__str__()+"\n")

    f_write_results.write("K = 9\n")
    f_write_results.write("P@9 = " + precision_9.__str__()+"\n")
    f_write_results.write("R@9 = " + recall_9.__str__()+"\n")
    f_write_results.write("F@9 = " + f1_9.__str__()+"\n")
    f_write_results.write("SR@9 = " + sr_9.__str__()+"\n")

    f_write_results.write("K = 10\n")
    f_write_results.write("P@10 = " + precision_10.__str__()+"\n")
    f_write_results.write("R@10 = " + recall_10.__str__()+"\n")
    f_write_results.write("F@10 = " + f1_10.__str__()+"\n")
    f_write_results.write("SR@10 = " + sr_10.__str__()+"\n")

    # f_write_results.write(prediction_not_seen)
    f_write_results.write("How many unseen labels:\n")
    f_write_results.write(str(len(prediction_not_seen))+"\n")
    f_write_results.write("How many unseen labels usage\n")
    sum = 0
    #f_write_results.write(json.dumps(prediction_not_seen)+"\n")
    for key, items in prediction_not_seen.items():
        sum += items
    f_write_results.write(str(sum)+"\n")
    f_write_results.write("\n")
    f_write_results.write("\n")
    # f_write_results.write(prediction_not_seen_correct)
    f_write_results.write("How many unseen labels correct:\n")
    f_write_results.write(str(len(prediction_not_seen_correct))+"\n")
    f_write_results.write("How many unseen labels used correctly\n")
    sum = 0
    for key, items in prediction_not_seen_correct.items():
        sum += items
    f_write_results.write(str(sum)+"\n")

    f_write_results.close()
