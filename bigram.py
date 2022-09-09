import sys
from collections import defaultdict
import re
import random
import math
import string

from pandas import test

### Retrieve the test sentence and smoothing parameterfrom the command line
test_sentence = sys.argv[1]
k = float(sys.argv[2])

### default vocab if not specified
std_latin = list(string.ascii_letters + string.whitespace + string.punctuation)

def bigramModel(char_list, vocab=std_latin):
    vocab.extend(['beg','end'])
    cond_probs = {}
    bigrams = [(char_list[i], char_list[i+1]) for i in range(len(char_list)-1)]
    bigrams.insert(0, ('beg', char_list[0]))
    bigrams.append((char_list[-1], 'end'))
    bigrams_dict = defaultdict(list)
    for key, v in bigrams:
        bigrams_dict[key].append(v) # creates a dictionary with the format {char: [chars that follow char]}
    for c in vocab:
        # cond_probs setup: P(c|cj) = cond_probs[c][cj]
        cond_probs[c] = {cj: (bigrams_dict[cj].count(c) + k) / (len(bigrams_dict[cj]) + (k * len(vocab))) for cj in vocab}
    #print(cond_probs)
    #print(bigrams_dict)
    return cond_probs    


def perplexity(test_text, bigram_mdl):
    log_sum = 0
    log_sum += math.log(bigram_mdl[test_text[0]]['beg'], 2) + math.log(bigram_mdl[test_text[-1]]['end'], 2)
    for i in range(len(test_text)-1):
        log_sum += math.log(bigram_mdl[test_text[i+1]][test_text[i]])
    perplex = math.pow(2, ((-1/len(test_text))*log_sum))
    return perplex

def main():
    ### load greenlandic and ilocano corpora
    greenland = open("greenlandic.txt", 'r')
    ilocano = open("ilocano.txt", 'r')

    ### split corpora into list of characters to ease calculations of bigram model
    greenland_split = [char for char in greenland.read()][1:-1]
    ilocano_split = [char for char in ilocano.read()][1:-1]
    
    greenland_bigram_model = bigramModel(greenland_split)
    ilocano_bigram_model = bigramModel(ilocano_split)
    
    greenland_perplexity = perplexity(test_sentence, greenland_bigram_model)
    ilocano_perplexity = perplexity(test_sentence, ilocano_bigram_model)
    
    print("perplexity from kalaallisut model: ")
    print(greenland_perplexity)
    print("\nfrom ilocano model: ")
    print(ilocano_perplexity)
    predicted_language = "kalaallisut" if greenland_perplexity < ilocano_perplexity else "ilocano"
    print("\nPredicted language: %s\n" % predicted_language)
    greenland.close()
    ilocano.close()
    

if __name__ == "__main__":
    main()