import os
import re
import math

pos_folder = "./aclImdb/train/pos"
neg_folder = "./aclImdb/train/neg"
test_folder_neg = "./aclImdb/test/neg"
test_folder_pos = "./aclImdb/test/pos"
pos_dict = {}
neg_dict = {}


def split_tokens(file):
    tokens = re.findall(r"\b\w+\b|\!", open(file, "r", encoding="utf-8").read())
    return [token.lower() for token in tokens]


def fill_dict(folder, dictionary):
    length = 0
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        if os.path.isfile(f):
            data = split_tokens(f)
            document_dict = {}
            for word in data:
                # binary naive Bayes implementation
                if word in document_dict:
                    continue
                document_dict[word] = True

                length += 1
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1  # add one smoothing
    rare_words = []
    for word in dictionary:
        if dictionary[word] == 1:
            rare_words.append(word)
        else:
            dictionary[word] = math.log(dictionary[word] / length)

    for unk in rare_words:
        del dictionary[unk]
    # below is a problem if there were no rare words but Im assuming there are rare words
    dictionary["UNKNOWNTOKEN444"] = math.log(len(rare_words) / length)


def assign_points(dict, word, score):
    if word in dict:
        score += dict[word]
    else:
        score += dict["UNKNOWNTOKEN444"]
    return score


def test_data(test_folder, pos_dictionary, neg_dictionary):
    positive = len(pos_dictionary)
    negative = len(neg_dictionary)
    length = positive + negative
    positive = math.log(positive / length)
    negative = math.log(negative / length)
    count_neg = 0
    count_pos = 0
    for filename in os.listdir(test_folder):
        neg_score = 0
        pos_score = 0
        f = os.path.join(test_folder, filename)
        if os.path.isfile(f):
            data = split_tokens(f)
            for word in data:
                pos_score = assign_points(pos_dictionary, word, pos_score)
                neg_score = assign_points(neg_dictionary, word, neg_score)
        pos_score = math.exp(pos_score + positive)
        neg_score = math.exp(neg_score + negative)
        if neg_score >= pos_score:
            count_neg += 1
        else:
            count_pos += 1
    print(count_pos, count_neg)


fill_dict(pos_folder, pos_dict)
fill_dict(neg_folder, neg_dict)
test_data(test_folder_neg, pos_dict, neg_dict)
test_data(test_folder_pos, pos_dict, neg_dict)

