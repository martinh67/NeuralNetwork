# declare imports required
import time
import pandas as pd
import gensim
import pprint as pp
from gensim.models import KeyedVectors
from methods import build_neural_network, extract_comment_data, extract_submission_data, clean_text

# start the program time
start = time.time()

# main method
def main():

    # build a neural network model using the submission data
    model = build_neural_network(extract_submission_data(1000))

    # uncomment this line for the comment data model
    # model = build_neural_network(extract_comment_data("mnitdo"))

    # word to input into the model
    covid = "covid"

    # print a space
    print()

    # try
    try:
        # print the most similar words to covid from the network
        # using pprint so it is formatted nicely
        pp.pprint(model.wv.most_similar(covid))

    # error message
    except:
        print(f"{covid} could not be found")

    # print a space
    print()

    # print a space
    print()

    # print the terms closest to new variant that is not delta i.e., omicron
    pp.pprint(model.wv.most_similar(positive = ['new', 'variant'], negative = ['delta']))

    # print a space
    print()

    # get the word vectors
    word_vectors = model.wv

    # save the model
    word_vectors.save('word2vec.wordvectors')

    # load the word vector model
    wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')

    # get the vector of covid
    vector = wv[covid]

    # print a space
    print()

    # print the vector for covid using pprint
    pp.pprint(vector)

    # print a space
    print()

    # get a list of words in the model
    words = list(model.wv.index_to_key)

    # print a space
    print()

    # print the list of words in the model
    print(words)

    # print a space
    print()

    # some synonyms that are associated with covid
    covid_terms = ['coronavirus', 'omicron', 'virus', 'corona', 'delta']

    # try
    try:

        # for each of the covid terms in the list
        for covid_term in covid_terms:

            # print the similarity between covid and corona
            print()
            print("similarity between covid : " + covid_term + " = " + str(model.wv.similarity('covid', covid_term)))
            print()

    # handle any terms not part of the corpus
    except:

        # print error message
        print("term not available")

    # print a space
    print()

    # print the odd word out from the list
    print(model.wv.doesnt_match(['vaccine', 'virus', 'study']))

    # print a space
    print()

# magic method to run main function
if __name__ == "__main__":
    main()

# print time of the program
print("\n"  +40*"#")
print(time.time() - start)
print (40*"#")
