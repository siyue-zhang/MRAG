import re
import json
import string
import argparse
import unicodedata
from collections import Counter


COMMON_LOCATIONS = ['united states', 'india', 'united kingdom',
                    'australia', 'canada', 'china', 'new york', 'florida',
                    'japan', 'england', 'uk', 'germany', 'spain', 'georgia',
                    'france', 'new york city', 'alabama', 'russia',
                    'kentucky', 'ohio', 'us', 'california', 'texas',
                    'pakistan', 'mexico', 'new zealand', 'usa', 'israel']

# Eval normalization from DrQA
def normalize(s):
    """Normalize answer."""
    s = unicodedata.normalize("NFD", s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    s = white_space_fix(remove_articles(remove_punc(lower(s))))
    return s

def eval_temp(preds):
    em_counts = {'total': Counter(),
                 'static': Counter(),
                 'sampled': Counter(),
                 'start': Counter(),
                 'any': Counter()}

    for pred in preds:
        gold_answers = [normalize(a) for a in pred['answer']]
        date_type = pred['date_type']
        em_match = normalize(pred['pred_answer']) in gold_answers
        if date_type == 'orig':
            em_counts['static'][em_match] += 1
        elif date_type == 'start':
            em_counts['start'][em_match] += 1
        else:
            em_counts['sampled'][em_match] += 1
        em_counts['total'][em_match] += 1
        any_answer = [normalize(a) for a in pred['any_answer']]
        any_match = normalize(pred['pred_answer']) in any_answer
        em_counts['any'][any_match] += 1

    if sum(em_counts['static'].values())>0:
        print('{}: {} / {} = {:.1f}'.format(
            'static',
            em_counts['static'][True],
            sum(em_counts['static'].values()),
            100 * em_counts['static'][True] / sum(em_counts['static'].values())))
    
    if sum(em_counts['sampled'].values())>0:
        print('{}: {} / {} = {:.1f}'.format(
            'sampled',
            em_counts['sampled'][True],
            sum(em_counts['sampled'].values()),
            100 * em_counts['sampled'][True] / sum(em_counts['sampled'].values())))
        
    if sum(em_counts['start'].values())>0:
        print('{}: {} / {} = {:.1f}'.format(
            'start',
            em_counts['start'][True],
            sum(em_counts['start'].values()),
            100 * em_counts['start'][True] / sum(em_counts['start'].values())))
    
    if sum(em_counts['total'].values())>0:
        print('{}: {} / {} = {:.1f}'.format(
            'total',
            em_counts['total'][True],
            sum(em_counts['total'].values()),
            100 * em_counts['total'][True] / sum(em_counts['total'].values())))
    
    if sum(em_counts['any'].values())>0:
        print('{}: {} / {} = {:.1f}'.format(
            'any',
            em_counts['any'][True],
            sum(em_counts['any'].values()),
            100 * em_counts['any'][True] / sum(em_counts['any'].values())))

def main(args):
    with open(args.preds_path, 'r') as f:
        preds = [json.loads(l) for l in f]
    eval_temp(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='desc')
    parser.add_argument('preds_path', type=str,
                        help='')
    args = parser.parse_args()
    main(args)
