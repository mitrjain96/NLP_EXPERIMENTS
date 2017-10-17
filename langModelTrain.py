import custom_utils
import nltk
import numpy as np
import sys
import itertools
import lang_model
# Creating training matrices
def read_dataset():
    vocab_size=int(sys.argv[2])
    start_tok="STRT_TOK"
    end_tok="END_TOK"
    UknkownToken="Not_in_vocab"
    filename="/home/mit/Desktop/NLP_EXPERIMENTS/Sonnets.txt"
    with open( filename,'r') as file1:
        raw_dataset=file1.read().split('\n')
    training_dataset=[]
    sent=[]
    for line in raw_dataset:
        line=line.strip()
        if(line is "" or custom_utils.isInt(line)==True):
            continue
        sent=[]
        sent.append(start_tok)
        sent.extend(nltk.word_tokenize(line))
        sent.append(end_tok)
        # print(sent)
        training_dataset.append(sent)
    word_freq = nltk.FreqDist(itertools.chain(*training_dataset))
    vocab=word_freq.most_common(vocab_size-1)
    index_to_word=[ x[0] for x in vocab ]
    index_to_word.append(UknkownToken)
    word_to_index=dict([(w,i) for i,w in enumerate(index_to_word)])
    print(word_to_index)

    for i,sent in enumerate(training_dataset):
        training_dataset[i]=[w if w in word_to_index else UknkownToken for w in sent]
    train_x=np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in training_dataset])
    train_y=np.asarray([[word_to_index[w] for w in sent[1:]] for sent in training_dataset])
    return train_x,train_y

def main():

    train_x,train_y=read_dataset()
    hid_dim=int(sys.argv[1])
    vocab_size=int(sys.argv[2])

    print(np.array(train_x).shape)
    print(train_x)

    langMOdel= lang_model.RNN(hid_dim,vocab_size)
    langMOdel.trainX(train_x,train_y)
    # print("Predicting New Sentences:")
    lanModel.predictX()
    print("Training completed")


if __name__ =="__main__":
    main()
