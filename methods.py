# declare imports required
import creds
import praw
import pandas as pd
import re
from nltk.corpus import stopwords
import gensim

# obtain a unique list of english stopwords
stop = set(stopwords.words("english"))

# set up Reddit login credentials
reddit = praw.Reddit(client_id = creds.client_id,
client_secret = creds.client_secret, user_agent = creds.user_agent,
username = creds.username, password = creds.password)

# method to extract the comment data from a submission using the id
def extract_comment_data(id):

    # set the submission to the id
    submission = reddit.submission(id = id)

    # declare a list to hold the comment ids
    comment_id_list = []

    # declare a list to hold the body text of the comment
    text_list = []

    # for all of the top level comments in the submission
    for top_level_comment in submission.comments:

        # if the instance is top level comment
        if isinstance(top_level_comment, MoreComments):

            # continue the iteration
            continue

        # append the comment id to the list
        comment_id_list.append(top_level_comment.id)

        # append the body text to the list
        text_list.append(top_level_comment.body)

    # create a dataframe to hold the data
    df = pd.DataFrame({"comment id" : comment_id_list, "text" : text_list})

    # return the dataframe
    return df


# extract submission data from the coronavirus reddit
def extract_submission_data(number):

    # hot is popular and relatiely new
    subreddit = reddit.subreddit("coronavirus").hot(limit = number)

    # list to hold the id of a submission
    submission_id_list = []

    # list to hold the title of the submission
    text_list = []

    # for each submission within the subreddit
    for submission in subreddit:

        # append the id to a list
        submission_id_list.append(submission.id)

        # append the title to a list of the text
        text_list.append(submission.title)

    # put the ids in a dataframe
    df = pd.DataFrame({"submission_id" : submission_id_list,
     "text" : text_list})

    # return the dataframe
    return df


# method to clean the text
def clean_text(row):

    # declare a list to hold each sentence
    sentence = []

    # for each term in the row
    for term in row.split():

        # replace all nonletters with blank space and make all letters lowercase
        term = re.sub('[^a-zA-Z]', "", term.lower())

        # remove all single characters
        term = re.sub(r'\s+[a-zA-Z]\s+', '', term)

        # additionall preprocessing step to remove empty spaces
        term = re.sub(r'\s+','', term)

        # additional preprocessing step to remove puncation
        term = re.sub(r'\W','', term)

        # if the term has greater than 3 characters
        if len(term) >= 3:

            # append the term to the sentence
            sentence.append(term)

    # get rid of stop words in the sentence
    sentence = [word for word in sentence if word not in stop]

    # return a string with a space between them
    return " ".join(sentence)

# function to build the neural network model
def build_neural_network(df):

    # clean the text and add to a new column in the dataframe
    df['processed_text'] = df.text.apply(clean_text)

    # create the neural network model using Word2Vec
    model = gensim.models.Word2Vec(window = 5, min_count = 2, workers = 4)

    # apply gensims preprocessing to the text in order to tokenise the data
    processed_text = df.processed_text.apply(gensim.utils.simple_preprocess)

    # build the vocabulary required for the neural network
    model.build_vocab(processed_text, progress_per = 100)

    # train the model using the corpus
    model.train(processed_text, total_examples = model.corpus_count,
    epochs = model.epochs)

    # return the neural network model
    return model
