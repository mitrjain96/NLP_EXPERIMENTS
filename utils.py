import requests
import numpy as np
import os


def fetch_wikis():
    with open('wiki_links.txt','r') as f:
        for line in f:
            link = line.strip()


def fetch_wiki(url):
    response = requests.get(url)
    soup = bs(response.content,'html.parser')
    print(soup.get_text())
    return


def init_babi(fname):
    print("==> Loading test from %s" % fname)
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": ""} 
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:
            task["C"] += line
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            tasks.append(task.copy())

    return tasks

def init_babi_deploy(fname,query):
	task = {'C':"",'Q':query}
	with open(fname,'r') as f:
		for line in f:
			line = line.strip()
			line = line.replace('.',' . ')
			task['C'] += line
			task['C'] += " "
	tasks = []
	tasks.append(task.copy())
	return tasks



def get_babi_raw(id, test_id):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    if (test_id == ""):
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en/%s_train.txt' % babi_name))
    babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en/%s_test.txt' % babi_test_name))
    return babi_train_raw, babi_test_raw


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "one_hot":
        raise Exception("to_return = 'one_hot' is not implemented yet")


def load_glove(dim = 50):
    glove = {}
    # path = "/Users/meshde/Mehmood/EruditeX/data/glove/glove.6B.50d.txt"
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/glove/glove.6B.%sd.txt' % dim)
    with open(path,'r') as f:
        for line in f:
            l = line.split()
            glove[l[0]] = list(map(float,l[1:]))
    return glove


def get_vector(word,glove):
    try:
        ans = np.array(glove[word]).reshape((1,50))
        return ans
    except:
        return np.random.rand(1,50)


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)


def main():
    url = "https://en.wikipedia.org/wiki/Stanford_University"
    fetch_wiki(url)
    return


if __name__ == '__main__':
    main()
            