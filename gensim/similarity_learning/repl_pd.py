import re
import pandas as pd
from pprint import pprint

def preprocess(sentence):
    return re.sub("[^a-zA-Z0-9]", " ", sentence.lower())



file_path = 'data/WikiQACorpus/WikiQA-train.tsv'

with open(file_path, encoding='utf8') as f:
    df = pd.read_csv(f, sep='\t')


# print(df)

questions = []
for Question, Answer in df.groupby('QuestionID').apply(dict).items():
    document_group = []
    for q, d, l in zip(Answer['Question'], Answer['Sentence'], Answer['Label']):
        document_group.append([preprocess(q), preprocess(d), l])
    
    questions.append(document_group)

pprint(questions)
print(len(questions))

print("-----------------------------------------------------------------")

QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
ANSWER_INDEX = 5
LABEL_INDEX = 6

with open(file_path, encoding='utf8') as f:
    lines = []
    for line in f:
        lines.append(line.split('\t'))

    big_document_group = []
    document_group = []
    for i, line in enumerate(lines[1:]):
        i += 1
        if i+1 < len(lines):
            if lines[i][QUESTION_ID_INDEX] == lines[i+1][QUESTION_ID_INDEX]:
                document_group.append(
                    [preprocess(lines[i][QUESTION_INDEX]),
                    preprocess(lines[i][ANSWER_INDEX]), int(lines[i][LABEL_INDEX])])
            else:
                document_group.append(lines[i][QUESTION_INDEX])
                big_document_group.append(document_group)
                document_group = []
        else:
            document_group.append(lines[i][QUESTION_INDEX])
            big_document_group.append(document_group)

pprint(big_document_group)
print(len(big_document_group))
            