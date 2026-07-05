import re
import collections
import torch
import unicodedata

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Clean and normalize input text
def clean_str(string, nmt=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if nmt:
        return unicodeToAscii(string.strip().lower())
    else:
        return string.strip().lower()
    

def load_snips_data(seq_file, label_file, label_dictionary):

    with open(seq_file, "r", encoding="utf-8") as f:
        x_text = [clean_str(line.strip()) for line in f.readlines()]

    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    y = []
    for label in labels:
        if label not in label_dictionary:
            label_dictionary[label] = len(label_dictionary)
        y.append(label_dictionary[label])

    return x_text, y, label_dictionary


def load_mr_data(pos_file, neg_file):
    pos_text = list(open(pos_file, "r", encoding='latin-1').readlines()) 
    pos_text = [clean_str(sent) for sent in pos_text] 

    neg_text = list(open(neg_file, "r", encoding='latin-1').readlines())
    neg_text = [clean_str(sent) for sent in neg_text]

    positive_labels = [1 for _ in pos_text] 
    negative_labels = [0 for _ in neg_text] 
    y = positive_labels + negative_labels

    x_final = pos_text + neg_text
    return x_final, y


# Build vocabulary from training sentences
def buildVocab(sentences, vocab_size):
    words = []
    for sentence in sentences:
        words.extend(sentence.split()) 
    print("The number of words: ", len(words))
    word_counts = collections.Counter(words)
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} 
    return vocabulary, vocabulary_inv


def text_to_indices(x_text, word_id_dict, use_unk=False):
    text_indices = []

    for text in x_text:
        words = text.split()
        ids = [2]  # <s>
        for word in words: 
            if word in word_id_dict:
                word_id = word_id_dict[word]
            else:  
                if use_unk:
                    word_id = 1 
                else:
                    word_id = len(word_id_dict)
                    word_id_dict[word] = word_id
            ids.append(word_id) 
        ids.append(3)  # </s>
        text_indices.append(ids)
    return text_indices


def sequence_to_tensor(sequence_list, nb_paddings=(0, 0)):
    nb_front_pad, nb_back_pad = nb_paddings

    max_length = len(max(sequence_list, key=len)) + nb_front_pad + nb_back_pad
    sequence_tensor = torch.LongTensor(len(sequence_list), max_length).zero_() 
    print(f"\nMax sequence length: {max_length}")
    for i, sequence in enumerate(sequence_list):
        sequence_tensor[i, nb_front_pad:len(sequence) + nb_front_pad] = torch.tensor(sequence)
    return sequence_tensor