from sklearn.model_selection import StratifiedKFold, cross_val_score
import sentence_transformers
import pandas as pd
from groq import Groq
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
print("-------------------------------------------------")
df = pd.read_csv("data.csv")
texts = df['text'].tolist()
labels = df['label'].tolist()

print("Loading SBert Model....")
sbert_model = SentenceTransformer("all-mpnet-base-v2")
embeddings = sbert_model.encode(texts,show_progress_bar=True)
print("----------------------------------------------------")
clf = LogisticRegression(max_iter=1000)
print("----------------------------------------------------")
clf.fit(embeddings,labels)
print("------------------------------------------------------")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring='accuracy')
print(f"Cross-validation Accuracy: {scores.mean():.4f}")


