from __future__ import print_function
from nltk.stem import *
from nltk.corpus import stopwords
import pandas as pd
import re
import collections
import cPickle
import random
import numpy as np
import tensorflow as tf
import matplotlib

# Force matplotlib to not use any Xwindows backend. # useful in Bash for Ubuntu on Windows 10
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

size_most_common_words = 20000  # size of the dictionnary

voc_size = size_most_common_words + 2  # size of the dictionnary + UNK + BGFILE (to separate words and understand that
# some of them are often at the beginning or the end of a sentence

embedding_size = 24  # dimensions of the vector representing an embeding
num_sampled = 64  # Number of negative examples to sample.
number_of_step = 100000  # Number of step
window_size = 2  # where to look
batch_size = 128  # number of sample per batch
plot_only = 2000  # number of words to plot
version = "1"  # to create new batches of examples without deleting old element in the folder


# Read the original file where the data was saved
def read_file(filename='../data/Combined_News_DJIA.csv'):
    data = pd.read_csv(filename)
    return data


# Tokenize a text
def tokenizer(text):
    # Remove english stop words
    stop = stopwords.words('english')
    # Keep track of emoticones
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    # '\W' retire tous les non mots + passe tout en miniscule + ajoute les emoticones sans le nez - (:-) -> :) )
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    # Split word based on space
    tokenized = [w for w in text.split() if w not in stop]
    #  Transform each word into into its lexical root
    stemmer = PorterStemmer()
    tokenized = [stemmer.stem(w) for w in tokenized]
    # Delete the b at each beginning of a sentences  (proper to Combined_News_DJIA.csv
    tokenized = tokenized[1:]
    return tokenized


# Transform the original sentences into vector of words
def tokenizing_file():
    data = read_file()  # read the file

    # Add a line to combine all headlines
    data['Combined'] = data.iloc[:, 2:27].apply(lambda row: ''.join(str(row.values)), axis=1)

    # tokenize all entries
    print('Tokenize and clean the file')
    for column in data:
        if column != 'Date' and column != 'Label':
            for index, _ in data.iterrows():
                data.set_value(index, column, tokenizer(str(data[column][index])))

    # save the new tokens
    # data.to_csv('transform.csv')
    data.to_pickle('tokenized_sentences' + version + '.p')
    print('saved tokens')


# Create a dictionnary
def create_dictionnary(data,
                       size_most_common_words=size_most_common_words):
    all_words = []
    for words in data['Combined']:
        for word in words:
            all_words.append(word)

    print("Number of words ", len(all_words))
    # Keep only the most seem words up to 3000

    count = collections.Counter(all_words).most_common(size_most_common_words)

    print('Creating dictionnaries')
    # Create index to word dictionnaries
    rdic = []
    # For words that are not the most seen
    rdic.append('UNK')
    # For appending at the end of each file
    rdic.append('BGFILE')
    for i in count:
        rdic.append(i[0])
    cPickle.dump(rdic, open('rdic' + version + '.p', 'wb'))

    # Create word to index dictionnaries
    dic = {}
    for i in range(len(rdic)):
        dic[rdic[i]] = i
    cPickle.dump(dic, open('dic' + version + '.p', 'wb'))
    windows_size = 2
    for column in data:
        if column != 'Date' and column != 'Label':
            for index, _ in data.iterrows():
                words = data[column][index]
                transformData = []
                for _ in range(windows_size):
                    transformData.append(1)
                for word in words:
                    if word in dic:
                        index1 = dic[word]
                    else:
                        index1 = 0
                    transformData.append(index1)
                for _ in range(windows_size):
                    transformData.append(1)
                data.set_value(index, column, transformData)
    # save as csv and p
    data.to_pickle('word_as_number' + version + '.p')
    data.to_csv("word_as_number" + version + ".csv")
    print('Dictionnary created')


def generate_batch(data,
                   size,
                   window_size,
                   debug=0):
    skip_gram_pair = []
    x_data = []
    y_data = []
    for i in range(size):
        cbow_pairs = []
        words = data['Top' + str(random.randint(1, 25))][random.randint(0, data.shape[0] - 1)]
        # print("size of words", len(words))
        if len(words) - window_size - 1 < window_size:
            i = i - 1
            continue
        index = random.randint(window_size, len(words) - window_size - 1)
        cbow_pairs.append([[words[index + k] for k in range(-window_size, window_size + 1) if k != 0], words[index]])
        for pair in cbow_pairs:
            for w in pair[0]:
                skip_gram_pair.append([pair[1], w])
    r = np.random.choice(range(len(skip_gram_pair)), size, replace=False)
    for i in r:
        x_data.append(skip_gram_pair[i][0])
        y_data.append([skip_gram_pair[i][1]])
    return x_data, y_data

# Plot the data
def plot_with_labels(final_embeddings, filename='tsne' + version + '.png'):
    reverse_dictionary = cPickle.load(open('rdic' + version + '.p', 'rb'))
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(70, 70))  # dimension of the images (larger is more readable)
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)



def build_model(data):
    print('Building model')
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_label = tf.placeholder(tf.int32, shape=[batch_size, 1])

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Xavier init for weights
    fan_in = voc_size
    fan_out = embedding_size
    low = -4 * np.sqrt(6.0 / (fan_in + fan_out))  # 4 for sigmoid
    high = 4 * np.sqrt(6.0 / (fan_in + fan_out))
    nce_weight = tf.Variable(tf.random_uniform([voc_size, embedding_size], minval=low, maxval=high, dtype=tf.float32))
    # init bias
    nce_biases = tf.Variable(tf.zeros([voc_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weight, nce_biases, embed, train_label, num_sampled, voc_size))

    train_op = tf.train.AdamOptimizer().minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # saved the mean nce_loss
    lossV = {}
    print('Starting session')
    with tf.Session() as sess:
        min_lost = 10000000.0
        tf.initialize_all_variables().run()

        for step in range(number_of_step):
            batch_inputs, batch_labels = generate_batch(data, batch_size, window_size)
            sess.run([train_op], feed_dict={train_inputs: batch_inputs, train_label: batch_labels})
            trained_embedding = embeddings.eval()
            if step % 500 == 0:
                loss_val = sess.run([loss], feed_dict={train_inputs: batch_inputs,
                                                       train_label: batch_labels})
                lossV[step] = loss_val[0]
                print("Iter %d: loss of %.5f" % (step, loss_val[0]))
                if min_lost > loss_val[0]:
                    print('Saving current state')
                    min_lost = loss_val
                    # Save the best model
                    np.save('word2vec' + version, trained_embedding)
                    # Save the best embeddings corresponding
                    final_embeddings = normalized_embeddings.eval()
                    # Save in file the best embedding
                    cPickle.dump(final_embeddings, open('final_embedding' + version + '.p', 'wb'))
    plot_with_labels(final_embeddings)


def sentences_to_embedding(embeddings, reverse_dictionnary):
    if embeddings is None:
        embeddings = cPickle.load(open('final_embedding' + version + '.p', 'rb'))
    if reverse_dictionnary is None:
        reverse_dictionnary = cPickle.load(open('rdic' + version + '.p', 'rb'))

    labels = {i: embeddings[i] for i in range(len(reverse_dictionnary))}
    cPickle.dump(labels, open('index_to_embedding' + version + '.p', 'wb'))

    labels = {reverse_dictionnary[i]: embeddings[i] for i in range(len(reverse_dictionnary))}
    cPickle.dump(labels, open('word_to_embedding' + version + '.p', 'wb'))


if __name__ == '__main__':
    # tokenizing_file()

    data = pd.read_pickle('tokenized_sentences' + version + '.p')
    create_dictionnary(data)

    build_model(data)



    sentences_to_embedding(None, None)
