import pickle
import math
import sys
import os.path

class Word_POS:

    delim = "/"
    def __init__(self, data, is_training):
        if is_training:
            split = data.split(self.delim)
            self.word = "/".join(split[:-1])
            self.tag = split[-1]
        else:
            self.word = data

class DataParser:
    def __init__(self, corpus_files):

        self.train_tagged, is_train_1 = corpus_files[0]

        self.train_sentences = []

        self._parse_file(self.train_tagged, self.train_sentences, is_training=is_train_1)

    def get_training_data(self):
        return self.train_sentences

    def _parse_file(self, filename, lst, is_training):
        if not filename or not os.path.isfile(filename):
            raise Exception("File not Found")

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                # Split the sentence into word_tag pieces. Considering space as the delimiter
                word_tag = line.split()

                # Convert string word/tag to Atom object
                word_POS = [Word_POS(data, is_training) for data in word_tag]

                # Add the atomised sentence to our list
                lst.append(word_POS)

class K_Fold_Cross_Validation:

    def __init__(self, k, training_data):

        self.portions = []

        num_sentences = int(len(training_data) / k)
        split = []

        for i, sent in enumerate(training_data):
            split.append(sent)
            if (( i + 1 ) % num_of_sentences == 0):
                self.portions.append(split)
                split = []
        if split:
            self.portions.append(split)

    def get_train_and_test_data(self, index=0):
        return self.portions[index], [portion for i, portion in enumerate(self.portions) if i != index], index+1


class HMM:
    start_tag = '$^START^$'

    def __init__(self, corpus_files):
        self.words_given_pos = {}
        self.pos3_given_pos2_and_pos1 = {}
        self.parser = DataParser(corpus_files)
        self.word_to_tag = {}
        self.word_tag_count = {}
        self.tag_count = {}
        self.trigrams = {}
        self.bigrams = {}
        self.tags = set()
        self.words = set()

        self.transition_backoff = {}
        self.emission_backoff = {}

        self.transition_singleton = {}
        self.emission_singleton = {}

        self.transition_one_count = {}
        self.emission_smoothed = {}

        self.num = 0

    def calculate_probabilities(self):
        self.populate_dictionaries()
        self.ProbWordGivenTag()
        self.ProbTrigramTags()
        self.BackoffProbabilities()
        self.SingletonCounts()
        self.SmoothedProbabilities()
        self._save()

    def populate_dictionaries(self):
        self.pos_tags = set()
        for sentence in self.parser.get_training_data():

            sentence.insert(0, Word_POS('$^START^$' + Word_POS.delim + '$^START^$', is_training=True ))
            sentence.insert(0, Word_POS('$^START^$' + Word_POS.delim + '$^START^$', is_training=True ))

            start_index = 2
            for i in range(start_index, len(sentence)):

                trigram_triplet = (( sentence[i - 2]).tag, (sentence[i - 1]).tag, (sentence[i]).tag )
                bigram_tuple = (( sentence[i - 2]).tag, (sentence[i - 1]).tag )
                self.trigrams[trigram_triplet] = self.trigrams.get(trigram_triplet, 0) + 1
                self.bigrams[bigram_tuple] = self.bigrams.get(bigram_tuple, 0) + 1

            for i, atom in enumerate(sentence):

                word = atom.word
                tag = atom.tag
                self.num += 1

                self.transition_backoff[tag] = self.transition_backoff.get(tag, 0) + 1
                self.emission_backoff[word] = self.emission_backoff.get(word, 0) + 1

                self.tags.add(tag)
                self.words.add(word)

                self.word_tag_count[ (word, tag) ] =  self.word_tag_count.get((word, tag), 0) + 1
                self.tag_count[ tag ] = self.tag_count.get(tag, 0) + 1
                if word not in self.word_to_tag:
                    self.word_to_tag[ word ] = set()
                self.word_to_tag[ word ].add(tag)

        print(self.bigrams)

        print(self.trigrams)

    def BackoffProbabilities(self):
        V = len(self.tags)
        print(self.num, V)
        for word in self.emission_backoff:
            self.emission_backoff[word] = float(1 + self.emission_backoff[word]) / float(self.num + V)

        for tag in self.transition_backoff:
            self.transition_backoff[tag] = float(self.transition_backoff[tag]) / float(self.num)

    def SingletonCounts(self):
        for i, tag_1 in enumerate(self.tags):
            for j, tag_2 in enumerate(self.tags):
                for k, tag_3 in enumerate(self.tags):
                    if i != j and i != k and j != k:
                        triplet = (tag_3, tag_2, tag_1)
                        if triplet in self.trigrams and self.trigrams[triplet] == 1:
                            self.transition_singleton[(tag_3, tag_2)] = self.transition_singleton.get((tag_3, tag_2), 0) + 1

        for word in self.words:
            for tag in self.tags:
                word_tag = (word, tag)
                if word_tag in self.word_tag_count and self.word_tag_count[word_tag] == 1:
                    self.emission_singleton[tag] = self.emission_singleton.get(tag, 0) + 1

    def SmoothedProbabilities(self):
        start_index = 2
        for sentence in self.parser.get_training_data():
            for i in range(start_index, len(sentence)):
                trigram_triplet = ( (sentence[i- 2]).tag , (sentence[i- 1]).tag, (sentence[i]).tag)
                bigram_tuple = ( (sentence[i- 2]).tag, (sentence[i- 1]).tag )
                lamda = self.transition_singleton.get(bigram_tuple, 0) + 1
                self.transition_one_count[trigram_triplet] = math.log(float(self.trigrams[trigram_triplet] + lamda * self.transition_backoff[sentence[i].tag]) / float(self.bigrams[bigram_tuple] + lamda))

        for word, tags_set in self.word_to_tag.items():
            for tag in tags_set:
                lamda = 1 + self.emission_singleton.get(tag, 0)
                self.emission_smoothed[(word, tag)] = math.log(float(self.word_tag_count[(word, tag)] + lamda * self.emission_backoff[word]) / float(self.tag_count[tag] + lamda))

    def _save(self):
        dictionaries = {"unique_tags" : self.tags, "bigram" : self.bigrams, "transmission" : self.pos3_given_pos2_and_pos1, "emission" : self.words_given_pos, "word2tag" : self.word_to_tag,
                        "transition_backoff" : self.transition_backoff, "emission_backoff" : self.emission_backoff,
                        "transition_singleton" : self.transition_singleton, "emission_singleton" : self.emission_singleton,
                        "transition_smoothed" : self.transition_one_count, "emission_smoothed" : self.emission_smoothed,
                        "tag_count" : self.tag_count, "n" : self.num}
        output = open('hmmmodel.txt', 'wb')
        pickle.dump(dictionaries, output)
        output.close()

    def ProbWordGivenTag(self):
        for word, tags_set in self.word_to_tag.items():
            for tag in tags_set:
                self.words_given_pos[(word, tag)] = math.log(float(self.word_tag_count[(word, tag)]) / float(self.tag_count[tag]))

    def ProbTrigramTags(self):
        start_index = 2
        V = len(self.tags)
        for sentence in self.parser.get_training_data():
            for i in range(start_index, len(sentence)):
                trigram_triplet = ((sentence[i- 2]).tag, (sentence[i- 1]).tag, (sentence[i]).tag)
                bigram_tuple = ((sentence[i- 2]).tag, (sentence[i- 1]).tag)
                self.pos3_given_pos2_and_pos1[trigram_triplet] = math.log(float(1 + self.trigrams[trigram_triplet]) / float(V + self.bigrams[bigram_tuple]))

def main(targets):
    # make the clean target
    if 'clean' in targets:
        print("Cleaning data")
        os.remove('./hmmmodel.txt')

    #downloads raw data
    if 'train' in targets:
        filename = sys.argv[2]
        hmm = HMM([(filename, True), (filename, True), (filename, True)])
        hmm.calculate_probabilities()

    # take small chunk of data
    if 'test' in targets:
        filename = sys.argv[2]
        hmm = HMM([(filename, False), (filename, False), (filename, False)])
        # hmm.load()
        # hmm.run()
        print("unable to decode currently, work in progress")
    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
