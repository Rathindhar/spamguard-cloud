# ============================================================
#  train_on_render.py — Trains the model during Render build
#  This runs ONCE during deployment to create combined_model.pkl
# ============================================================

import os, re, sys, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes     import MultinomialNB
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing   import MinMaxScaler
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(DATA_DIR, "combined_model.pkl")
CLEAN_PATH   = os.path.join(DATA_DIR, "cleaned_data.csv")
SMS_PATH     = os.path.join(DATA_DIR, "SMSSpamCollection")

print("=" * 55)
print("  SpamGuard — Training model on Render")
print("=" * 55)

# ── Stop words ────────────────────────────────────────────────
STOP = {
    "a","an","the","and","or","but","is","are","was","were","be","been",
    "have","has","had","do","does","did","will","would","may","might",
    "must","can","could","not","no","i","me","my","we","our","you","your",
    "he","she","it","its","they","them","this","that","to","of","in",
    "for","with","on","at","by","from","up","out","s","t","ll"
}

def clean(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in STOP and len(w) > 1)

# ── Build training dataset ────────────────────────────────────
print("\n  Building training dataset...")

# Core spam/ham training examples (Indian context aware)
TRAINING_DATA = [
    # Classic spam
    ("spam","Congratulations you won free iPhone click here now prize claim"),
    ("spam","URGENT claim prize call immediately free cash winner reward"),
    ("spam","Win dollars limited time offer click now free prize"),
    ("spam","Free ringtones claim prize now urgent money earn daily"),
    ("spam","Selected cash prize call bank account paytm kyc pending"),
    ("spam","Earn money daily home trick register secret method online"),
    ("spam","Paytm KYC pending update avoid suspension deactivated blocked"),
    ("spam","Limited loan offer zero interest apply today premium membership"),
    ("spam","Free Netflix subscription activate now months upgrade account"),
    ("spam","Government subsidy scheme selected congratulations laptop free"),
    ("spam","Hot investment opportunity guaranteed profits join crypto returns"),
    ("spam","Free recharge worth available claim fast sim card deactivated"),
    ("spam","Exclusive crypto returns 200 percent investment opportunity"),
    ("spam","Google account suspicious activity verify confirm now blocked"),
    ("spam","Click discount branded clothes get now download app earn"),
    ("spam","Secret online method earn money quickly daily income trick"),
    ("spam","Free laptop government student scheme apply now subsidy"),
    ("spam","Activate premium membership free today subscription upgrade"),
    ("spam","Email storage full upgrade immediately account suspended"),
    ("spam","Download app earn rewards daily income money register"),
    ("spam","Last chance cashback reward Rs 20000 claim now today"),
    ("spam","Win brand car lucky draw today participate register claim"),
    ("spam","Parcel delivery failed update address verify now http"),
    ("spam","Bank account blocked verify immediately http secure login"),
    ("spam","Free Amazon gift card worth Rs 5000 click here claim"),
    ("spam","iPhone 15 Rs 999 limited offer click buy now today"),
    # Indian specific spam
    ("spam","Earn Rs 8000 weekly from home no experience needed register"),
    ("spam","Your ATM card blocked update details verify now immediately"),
    ("spam","Win free trip Goa register today lucky draw prize claim"),
    ("spam","Get instant personal loan approval apply today zero interest"),
    ("spam","Your KYC incomplete update immediately avoid suspension"),
    ("spam","Your SIM deactivated soon act now update details verify"),
    ("spam","Earn passive income simple method daily online register"),
    ("spam","Free gift card waiting click now claim reward today"),
    ("spam","Get free data 30 days activate now mobile recharge"),
    ("spam","Limited offer get 80 percent off electronics buy now"),
    ("spam","Free insurance policy available today claim register now"),
    ("spam","Earn Rs 10000 monthly home easily trick method daily"),
    ("spam","Your Aadhaar update pending complete now verify immediately"),
    ("spam","Your bank account urgent verification update details now"),
    ("spam","Earn money watching videos online daily income app"),
    ("spam","Win free tickets Dubai register lucky draw prize claim"),
    ("spam","Get free access premium content upgrade account now"),
    ("spam","Your account suspended verify now blocked pending update"),
    ("spam","Earn daily income app download register money online"),
    ("spam","Click claim surprise gift reward free now today"),
    ("spam","Limited cashback offer available claim instantly today"),
    ("spam","Claim reward before expires cashback free today limited"),
    ("spam","Investment opportunity available now returns profit guaranteed"),
    ("spam","Click unlock premium features account membership free"),
    ("spam","PAN card verification required update details now immediately"),
    ("spam","Win exciting prizes registering today lucky draw claim"),
    ("spam","Voucher worth Rs 2000 claim free today limited offer"),
    ("spam","Instant loan approval apply now zero interest credit"),
    # Ham
    ("ham","Hey are we still meeting lunch today"),
    ("ham","Can you send notes from yesterday class"),
    ("ham","Mom asked if you can buy milk on way home"),
    ("ham","Happy birthday hope you have great day"),
    ("ham","Professor uploaded lecture slides exam"),
    ("ham","Can you help me debug this Java code"),
    ("ham","I just reached library today"),
    ("ham","Are you free this evening for call"),
    ("ham","I uploaded assignment to portal today"),
    ("ham","Can we reschedule meeting to tomorrow"),
    ("ham","Did you complete lab record today class"),
    ("ham","Can you explain this recursion problem"),
    ("ham","I pushed code to GitHub repository"),
    ("ham","Did you attend today lecture professor notes"),
    ("ham","Let us prepare for interview together coding"),
    ("ham","Can you help me with DBMS concepts exam"),
    ("ham","I reached home safely today evening"),
    ("ham","Let us revise OS concepts together tonight exam"),
    ("ham","Lecture was really helpful today professor"),
    ("ham","I updated project documentation files today"),
    ("ham","Can you review my resume for interview"),
    ("ham","I fixed bug in code today java"),
    ("ham","Can you explain polymorphism again class notes"),
    ("ham","Seminar starts at 2 PM today college"),
    ("ham","Let us practice coding problems together tonight"),
    ("ham","I completed Java assignment today portal"),
    ("ham","Let us revise DSA together tonight exam"),
    ("ham","Did you finish homework today submission"),
    ("ham","I am preparing for tomorrow exam revision"),
    ("ham","Class has been postponed today professor"),
    ("ham","I am attending seminar now college"),
    ("ham","Results will be out next week portal"),
    ("ham","Let us go for lunch at 1 PM canteen"),
    ("ham","I will share notes later group"),
    ("ham","I completed recursion problems practice"),
    ("ham","Let us plan for weekend trip friends"),
    ("ham","I sent email to professor today"),
    ("ham","I fixed bug in code today"),
    ("ham","Let us meet after college today"),
    ("ham","I will send file shortly email"),
    ("ham","I completed practice problems today"),
    ("ham","Let us go for walk evening"),
    ("ham","I will help you with coding later"),
    ("ham","Meeting link shared in group"),
    ("ham","I completed recursion problems today"),
    ("ham","Can we discuss project tomorrow"),
    ("ham","I am attending seminar now"),
    ("ham","The results will be announced tomorrow"),
    ("ham","I will share github repository link"),
    ("ham","Good luck for interview tomorrow"),
]

# Repeat each example to give enough weight in training
rows = []
for label, msg in TRAINING_DATA:
    for _ in range(10):
        rows.append({"label": label, "label_num": 1 if label=="spam" else 0,
                     "clean_message": clean(msg)})

# Load SMS dataset if available for better generalisation
if os.path.exists(SMS_PATH):
    print(f"  Loading SMS Spam Collection...")
    df_sms = pd.read_csv(SMS_PATH, sep="\t", header=None,
                         names=["label","message"], encoding="latin-1")
    df_sms["label_num"]     = df_sms["label"].map({"spam":1,"ham":0})
    df_sms["clean_message"] = df_sms["message"].apply(clean)
    rows_sms = df_sms[["label","label_num","clean_message"]].to_dict("records")
    rows.extend(rows_sms)
    print(f"  SMS dataset: {len(df_sms)} messages added")

df = pd.DataFrame(rows)
df.to_csv(CLEAN_PATH, index=False)
print(f"  Total training data: {len(df)} rows")
print(f"  Spam: {df['label_num'].sum()}  Ham: {(df['label_num']==0).sum()}")

# ── TF-IDF features ───────────────────────────────────────────
print("\n  Building TF-IDF features...")
X_text = df["clean_message"].fillna("")
y      = df["label_num"].values

X_train_t, X_test_t, y_train, y_test = train_test_split(
    X_text, y, test_size=0.20, random_state=42, stratify=y
)
vectorizer    = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train_t)
X_test_tfidf  = vectorizer.transform(X_test_t)

# ── Behavioral features ───────────────────────────────────────
SPAM_KW = {
    "free","win","winner","won","prize","claim","urgent","congratulations",
    "selected","offer","cash","reward","click","call","subscribe","guaranteed",
    "limited","exclusive","bonus","discount","loan","investment","earn",
    "account","suspended","verify","confirm","bank","activate","download",
    "register","apply","update","upgrade","recharge","cashback","membership",
    "subscription","scheme","opportunity","returns","daily","instantly",
    "deactivated","blocked","pending","suspicious","paytm","kyc","sim",
    "crypto","government","subsidy","netflix","amazon","iphone","parcel",
    "delivery","lucky","draw","vacation","dubai","laptop","mobile","storage",
    "money","profits","join","premium","secret","online","method","trick",
    "fast","quickly","suspension","interest","zero","google","activity",
    "branded","clothes","worth","available","student","full","card","app",
}
HAM_KW = {
    "professor","lecture","slides","assignment","exam","class","notes",
    "study","college","coding","debug","java","github","repository",
    "recursion","interview","cricket","practice","library","birthday",
    "dinner","movie","charger","meeting","office","project","files",
    "results","announced","milk","home","bring","reach","lunch","buy","mom",
}

def get_features(text):
    if not text or len(text.strip()) == 0:
        return [0]*5
    words = [w.strip("!.,?:()").lower() for w in text.split()]
    spam_hits = sum(1 for w in words if w in SPAM_KW)
    ham_hits  = sum(1 for w in words if w in HAM_KW)
    url_count = len(re.findall(r"http|www\.|\.com", text, re.I))
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text),1)
    exclaim    = text.count("!")
    return [max(0, spam_hits-ham_hits), ham_hits, url_count, caps_ratio, exclaim]

print("  Extracting behavioral features...")
all_feats  = [get_features(t) for t in X_text]
feat_arr   = np.array(all_feats, dtype=float)
feat_arr   = np.nan_to_num(feat_arr)

scaler     = MinMaxScaler()
feat_scaled = scaler.fit_transform(feat_arr)

train_idx  = list(range(len(X_train_t)))
test_idx   = list(range(len(X_train_t), len(X_train_t)+len(X_test_t)))

# Re-split behavioral features to match text split
X_train_t_list = list(X_train_t)
all_texts      = list(X_text)

# Get indices properly
train_indices = X_train_t.index.tolist()
test_indices  = X_test_t.index.tolist()

X_train_b = feat_scaled[train_indices]
X_test_b  = feat_scaled[test_indices]

from scipy.sparse import hstack, csr_matrix
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_b)])
X_test_combined  = hstack([X_test_tfidf,  csr_matrix(X_test_b)])

# ── Train classifiers ─────────────────────────────────────────
print("\n  Training classifiers...")
classifiers = {
    "Naive Bayes"   : MultinomialNB(),
    "Logistic Reg"  : LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest" : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"           : CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
}

results = {}
for name, clf in classifiers.items():
    if "Naive" in name:
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
    else:
        clf.fit(X_train_combined, y_train)
        y_pred = clf.predict(X_test_combined)

    acc  = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec  = recall_score(y_test, y_pred, zero_division=0) * 100
    f1   = f1_score(y_test, y_pred, zero_division=0) * 100
    results[name] = {"Accuracy":acc,"Precision":prec,"Recall":rec,"F1-Score":f1}
    print(f"    {name:<20} Acc={acc:.1f}% F1={f1:.1f}%")

# ── Simple ensemble (best single model) ──────────────────────
best_name = max(results, key=lambda n: results[n]["F1-Score"])
best_clf  = classifiers[best_name]
print(f"\n  Best model: {best_name}")

# ── Drift detector (simple) ───────────────────────────────────
class SimpleDriftDetector:
    def __init__(self): self.n=0; self.errors=0
    def update(self, correct): self.n+=1; self.errors+=(0 if correct else 1)
    def drift_detected(self): return (self.errors/max(self.n,1))>0.15
    def status(self):
        rate = self.errors/max(self.n,1)*100
        if rate>15: return f"🔴 DRIFT DETECTED ({rate:.1f}% error)"
        elif rate>10: return f"🟡 WARNING: {rate:.1f}% error rate"
        return f"🟢 Stable ({rate:.1f}% error rate)"

drift = SimpleDriftDetector()

# ── Save model ────────────────────────────────────────────────
save = {
    "classifiers"        : classifiers,
    "best_name"          : best_name,
    "vectorizer"         : vectorizer,
    "behavioral_scaler"  : scaler,
    "drift_detector"     : drift,
    "results"            : results[best_name],
    "all_results"        : results,
    "feature_names"      : ["spam_score","ham_score","url_count","caps_ratio","exclaim"],
    "spam_keywords"      : SPAM_KW,
    "ham_keywords"       : HAM_KW,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(save, f)

print(f"\n  Model saved to: {MODEL_PATH}")
print(f"  Accuracy : {results[best_name]['Accuracy']:.1f}%")
print(f"  F1-Score : {results[best_name]['F1-Score']:.1f}%")
print("\n  Training complete!")