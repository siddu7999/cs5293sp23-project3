import pypdf
import normalize
import os
import json
import argparse
import pandas as pd
from sklearn.metrics import silhouette_score as sil_score
from sklearn.metrics import calinski_harabasz_score as ch_score
from sklearn.metrics import davies_bouldin_score as db_score
from sklearn.cluster import KMeans
import pickle
from joblib import dump,load
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


def clean_pdf_files(filenames):
    data={}
    for file in filenames:
        city=file.split('.')[0]
        with open(file,"rb") as f:
            reader=pypdf.PdfReader(f)
    
            num_pages=len(reader.pages)
            text=''
            for i in range(num_pages):
                page=reader.pages[i]
                text += page.extract_text()
            data[city]=text
    df = pd.DataFrame(data.items(), columns=['city','Raw_text'])
    c = pd.DataFrame()
    c['city'] = df['city']
    c['Raw_text']=df['Raw_text']
    c['Clean_text']=normalize.normalize_corpus(df['Raw_text'])


    clf=load('model.joblib')
    vectorizer = load('vectorizer.joblib')
    vec_data = vectorizer.transform(c['Clean_text'])
    """ X_count = vectorizer.fit_transform(c['Clean_text'])
    clf.fit(X_count.toarray())
    clf.fit(X_count.shape) """
    pred_val = clf.predict(vec_data.toarray())
    c['clusterid']= pred_val[0]
    c.to_csv('smartcity_predictions.tsv', sep='\t', escapechar='\\')
    return [city],pred_val[0]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Clean PDF files and perform clustering')
    parser.add_argument('--document', nargs='*',help='PDF file names',required=True,action='append')
    args = parser.parse_args()
    result = clean_pdf_files(args.document[0])
    print(result[0],"cluster id:", result[1])

