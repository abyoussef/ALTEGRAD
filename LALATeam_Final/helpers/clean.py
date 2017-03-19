import re
import string
import nltk
from nltk import PerceptronTagger
from nltk.corpus import stopwords

def clean_text_simple(text, tagger, keep, stpwds, stemmer, remove_stopwords=True, pos_filtering=True, stemming=True):
    # convert to lower case
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    tokens = tokenization(text)
    if pos_filtering == True and len(tokens) > 0:
        tokens = pos_tagging(tokens, tagger, keep)
    if remove_stopwords:
        # remove stopwords
        tokens = [token for token in tokens if token not in stpwds]
    if stemming:
        # apply Porter's stemmer
        tokens = map(stemmer.stem, tokens)
    return tokens


def pos_tagging(tokens, tagger, keep):
    # apply POS-tagging
    tagged_tokens = tagger.tag(tokens)
    # retain only nouns and adjectives
    tokens = [item[0] for item in tagged_tokens if item[1] in keep]
    return tokens


def tokenization(text):
    punct = string.punctuation.replace('-', '')
    cond = '[' + re.escape(punct) + ']+'
    text = re.sub(cond, ' ', text)
    text = re.sub('(\s+-|-\s+)', ' ', text)
    # strip extra white space
    text = re.sub('-{2,}', ' ', text)
    text = re.sub('\s+', ' ', text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    tokens = text.split(' ')
    tokens = filter(lambda x: len(x) > 0, tokens)
    return tokens


def clean(X, col = 'body', cleaner = clean_text_simple, join = True):
    X_cleaned = X.copy()
    tagger = PerceptronTagger()
    keep = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR'])
    stpwds = set(stopwords.words('english'))
    stemmer = nltk.stem.PorterStemmer()
    X_cleaned[col] = X_cleaned[col].apply(lambda x: cleaner(x, tagger, keep, stpwds, stemmer))
    if join:
        X_cleaned[col] = X_cleaned[col].apply(lambda x: ' '.join(x))
    return X_cleaned