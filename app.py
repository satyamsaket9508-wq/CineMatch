"""
🎬 CINEMATCH — Movie Recommendation System (single-file Streamlit app)
Run:   pip install streamlit scikit-learn pandas numpy scipy requests python-dotenv
       streamlit run app.py
Needs in the same folder: df.pkl, indices.pkl, tfidf.pkl, tfidf_matrix.pkl, movies.csv (or tmdb_ids.pkl)
Optional: TMDB_API_KEY in a .env file, in Streamlit secrets (no key box in the UI).
"""
import hashlib
import html
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity

import History_db

st.set_page_config(page_title="CineMatch · Find your next obsession", page_icon="🎬", layout="wide")

BASE = Path(__file__).parent
History_db.init_db()
GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
          "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"]
MOODS = {"Action": "💥", "Adventure": "🧭", "Animation": "🧸", "Comedy": "😂", "Crime": "🕵️", "Drama": "🎭",
         "Fantasy": "🐉", "Horror": "👻", "Mystery": "🔎", "Romance": "💘", "Science Fiction": "🛸",
         "Thriller": "🔪", "War": "🪖", "Western": "🤠"}

# ═════════════════════════════ DATA + MODEL ═════════════════════════════
def _pk(name):
    with open(BASE / name, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    import warnings
    warnings.filterwarnings("ignore")
    df = _pk("df.pkl").reset_index(drop=True)
    if (BASE / "tmdb_ids.pkl").exists():
        df["tmdb_id"] = _pk("tmdb_ids.pkl").values
    else:  # fall back to the original csv (same row order as your notebook)
        df["tmdb_id"] = pd.read_csv(BASE / "movies.csv", usecols=["id"])["id"].values[: len(df)]
    matrix = _pk("tfidf_matrix.pkl")
    indices = _pk("indices.pkl")
    try:
        vocab = np.array(_pk("tfidf.pkl").get_feature_names_out())
    except Exception:
        vocab = None
    return df, matrix, indices, vocab


DF, MATRIX, INDICES, VOCAB = load_model()
TITLES = sorted(DF["title"].unique().tolist())


def _idx(title):
    """Row number for a title (a few titles appear twice in the data; take the first)."""
    v = INDICES[title]
    return int(v.iloc[0] if hasattr(v, "iloc") else v)


def get_movie(title):
    return DF.iloc[_idx(title)].to_dict() if title in INDICES else None


def recommend(title, n=12, min_rating=0.0, genres=None):
    idx = _idx(title)
    sims = cosine_similarity(MATRIX[idx], MATRIX).ravel()
    sims[idx] = -1
    mask = (DF["vote_average"] >= min_rating).to_numpy()
    for g in genres or []:
        mask &= DF["genres"].str.contains(g, regex=False).to_numpy()
    sims = np.where(mask, sims, -1)
    top = [i for i in np.argsort(sims)[::-1][:n] if sims[i] > 0]
    out = DF.iloc[top].copy()
    out["score"] = sims[top]
    recs = out.to_dict("records")
    for r, row_i in zip(recs, top):  # explainability: which words made these two movies similar?
        r["shared"] = shared_terms(idx, int(row_i))
    return recs


def shared_terms(i, j, k=3):
    if VOCAB is None:
        return []
    prod = MATRIX[i].multiply(MATRIX[j]).tocoo()
    if prod.nnz == 0:
        return []
    order = np.argsort(prod.data)[::-1][:k]
    return [str(VOCAB[prod.col[o]]) for o in order]


def dna(title, k=8):
    if VOCAB is None:
        return []
    row = MATRIX[_idx(title)].tocoo()
    order = np.argsort(row.data)[::-1][:k]
    return [str(VOCAB[row.col[o]]) for o in order]


def pick_genre(g, n=18):
    d = DF[(DF["genres"].str.contains(g, regex=False)) & (DF["vote_average"] >= 6.3)]
    return d.nlargest(n, "popularity").to_dict("records")


def trending(n=18):
    return DF[DF["vote_average"] >= 6.0].nlargest(n, "popularity").to_dict("records")


def top_rated(n=18):
    return DF[DF["popularity"] >= 20].nlargest(n, "vote_average").to_dict("records")


def hidden_gems(n=18):
    d = DF[(DF["vote_average"] >= 7.3) & (DF["popularity"].between(3, 25))]
    return d.nlargest(n, "vote_average").to_dict("records")


# ═════════════════════════════ TMDB ═════════════════════════════
@st.cache_resource
def key_store():
    return {"key": (os.getenv("TMDB_API_KEY") or "").strip()}


try:
    if not key_store()["key"] and "TMDB_API_KEY" in st.secrets:
        key_store()["key"] = str(st.secrets["TMDB_API_KEY"]).strip()
except Exception:
    pass


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def tmdb_movie(tmdb_id, key):
    if not key:
        return None
    params, headers = ({}, {"Authorization": f"Bearer {key}"}) if key.startswith("eyJ") else ({"api_key": key}, {})
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}", headers=headers, timeout=8,
                         params={**params, "append_to_response": "videos,credits", "language": "en-US"})
        if r.status_code != 200:
            return None
        d = r.json()
    except requests.RequestException:
        return None
    trailer = next((f"https://www.youtube.com/watch?v={v['key']}" for v in d.get("videos", {}).get("results", [])
                    if v.get("site") == "YouTube" and v.get("type") == "Trailer"), None)
    return {
        "poster": f"https://image.tmdb.org/t/p/w500{d['poster_path']}" if d.get("poster_path") else None,
        "backdrop": f"https://image.tmdb.org/t/p/w1280{d['backdrop_path']}" if d.get("backdrop_path") else None,
        "runtime": d.get("runtime"), "year": (d.get("release_date") or "")[:4], "tagline": d.get("tagline"),
        "cast": [c["name"] for c in d.get("credits", {}).get("cast", [])[:6]], "trailer": trailer,
    }


def tmdb_many(movies):
    key = key_store()["key"]
    ids = [int(m["tmdb_id"]) for m in movies]
    if not key:
        return {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(ids, ex.map(lambda i: tmdb_movie(i, key), ids)))


# ═════════════════════════════ STYLES ═════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;800&display=swap');
:root{--bg:#05050b;--card:#12121d;--a:#ff2e63;--b:#08d9d6;--c:#a066ff;--gold:#ffd166;--txt:#f4f4f8;--mut:#9a9ab0}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--txt)}
.stApp{background:var(--bg)}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0}
.block-container{padding-top:.6rem;max-width:1500px;position:relative;z-index:2}
section[data-testid="stSidebar"]{background:rgba(12,12,22,.9);backdrop-filter:blur(14px);border-right:1px solid rgba(255,255,255,.07)}
::-webkit-scrollbar{width:9px}::-webkit-scrollbar-thumb{background:linear-gradient(var(--a),var(--c));border-radius:9px}

/* aurora + grain + grid */
.aurora{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none}
.aurora i{position:absolute;border-radius:50%;filter:blur(90px);opacity:.5;mix-blend-mode:screen;animation:float 18s ease-in-out infinite alternate}
.aurora i:nth-child(1){width:55vw;height:55vw;left:-15vw;top:-20vw;background:#ff2e63;animation-duration:19s}
.aurora i:nth-child(2){width:48vw;height:48vw;right:-15vw;top:-10vw;background:#08d9d6;opacity:.32;animation-duration:23s}
.aurora i:nth-child(3){width:50vw;height:50vw;left:25vw;bottom:-28vw;background:#7a3cff;opacity:.45;animation-duration:27s}
.aurora::after{content:"";position:absolute;inset:0;background:
 linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px) 0 0/54px 54px,
 linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px) 0 0/54px 54px;
 mask-image:radial-gradient(ellipse at 50% 30%,#000 10%,transparent 75%)}
.grain{position:fixed;inset:-50%;z-index:1;pointer-events:none;opacity:.07;animation:grain 1s steps(6) infinite;
 background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>")}
@keyframes grain{0%{transform:translate(0,0)}25%{transform:translate(-3%,2%)}50%{transform:translate(2%,-3%)}75%{transform:translate(-2%,-2%)}100%{transform:translate(3%,3%)}}
@keyframes float{to{transform:translate(8vw,6vw) scale(1.25)}}

/* ticker */
.ticker{overflow:hidden;white-space:nowrap;border-top:1px solid rgba(255,255,255,.1);border-bottom:1px solid rgba(255,255,255,.1);
 background:rgba(255,255,255,.03);margin:0 0 1.1rem;padding:.45rem 0;mask-image:linear-gradient(90deg,transparent,#000 8%,#000 92%,transparent)}
.ticker div{display:inline-block;animation:scroll 70s linear infinite;font-family:'Bebas Neue',Impact,'Arial Narrow',sans-serif;letter-spacing:.18em;font-size:1rem;color:#c9c9dc}
.ticker b{color:var(--gold);font-weight:400}.ticker span{color:var(--a);margin:0 1.1rem}
@keyframes scroll{to{transform:translateX(-50%)}}

/* logo */
.logo{position:relative;font-family:'Bebas Neue',Impact,'Arial Narrow',sans-serif;font-size:clamp(3rem,7vw,5.6rem);letter-spacing:.16em;line-height:1;display:inline-block;
 background:linear-gradient(90deg,#ff2e63,#ffd166,#08d9d6,#a066ff,#ff2e63);background-size:300% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;
 animation:flow 7s linear infinite;filter:drop-shadow(0 0 24px rgba(255,46,99,.45))}
.logo::before,.logo::after{content:attr(data-t);position:absolute;inset:0;-webkit-background-clip:text;background-clip:text;color:transparent;opacity:0}
.logo::before{background:var(--b);animation:glitch1 6s infinite}.logo::after{background:var(--a);animation:glitch2 6s infinite}
@keyframes glitch1{0%,92%,100%{opacity:0;transform:none}93%{opacity:.9;transform:translate(-5px,2px);clip-path:inset(10% 0 55% 0)}95%{opacity:.9;transform:translate(4px,-2px);clip-path:inset(60% 0 8% 0)}97%{opacity:0}}
@keyframes glitch2{0%,92%,100%{opacity:0;transform:none}94%{opacity:.9;transform:translate(5px,-2px);clip-path:inset(40% 0 30% 0)}96%{opacity:.9;transform:translate(-4px,2px);clip-path:inset(5% 0 75% 0)}98%{opacity:0}}
@keyframes flow{to{background-position:300% 0}}
.tag{color:var(--mut);letter-spacing:.4em;font-size:.72rem;text-transform:uppercase;margin:.3rem 0 1.1rem}
.tag b{color:var(--b)}

/* stat strip */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin:1.2rem 0}
.stat{padding:1rem 1.2rem;border-radius:18px;background:linear-gradient(135deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.1);backdrop-filter:blur(10px);animation:rise .7s ease both}
.stat strong{display:block;font-family:'Bebas Neue',Impact,'Arial Narrow',sans-serif;font-size:2.3rem;letter-spacing:.06em;background:linear-gradient(90deg,var(--gold),var(--a));-webkit-background-clip:text;background-clip:text;color:transparent}
.stat span{color:var(--mut);font-size:.72rem;letter-spacing:.2em;text-transform:uppercase}
@media(max-width:800px){.stats{grid-template-columns:repeat(2,1fr)}}

/* mood portals */
.moods{display:flex;flex-wrap:wrap;gap:.7rem;margin:.4rem 0 1rem}
a.mood{display:inline-flex;align-items:center;gap:.5rem;padding:.65rem 1.15rem;border-radius:999px;font-weight:700;font-size:.88rem;text-decoration:none!important;color:#fff!important;
 background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.16);backdrop-filter:blur(8px);transition:.3s cubic-bezier(.2,.8,.2,1)}
a.mood .e{font-size:1.15rem;transition:.3s}
a.mood:hover{background:linear-gradient(135deg,var(--a),var(--c));border-color:transparent;transform:translateY(-4px) scale(1.07);box-shadow:0 12px 30px rgba(160,102,255,.5)}
a.mood:hover .e{transform:scale(1.4) rotate(-12deg)}

/* hero */
.hero{position:relative;border-radius:26px;overflow:hidden;min-height:470px;display:flex;align-items:flex-end;border:1px solid rgba(255,255,255,.12);
 box-shadow:0 40px 90px rgba(0,0,0,.7),0 0 0 1px rgba(255,46,99,.15),0 0 80px rgba(160,102,255,.18);margin:.8rem 0 1.4rem;animation:rise .7s ease both}
.hero .bg{position:absolute;inset:0;background-size:cover;background-position:center 20%;animation:drift 26s ease-in-out infinite alternate}
.hero .shade{position:absolute;inset:0;background:linear-gradient(90deg,rgba(5,5,11,.97) 8%,rgba(5,5,11,.72) 46%,rgba(5,5,11,.1)),linear-gradient(0deg,rgba(5,5,11,.95),transparent 55%)}
.hero .scan{position:absolute;inset:0;background:repeating-linear-gradient(0deg,rgba(255,255,255,.025) 0 1px,transparent 1px 3px);pointer-events:none}
.hero .body{position:relative;padding:2.6rem 2.8rem;max-width:800px}
.hero h1{font-family:'Bebas Neue',Impact,'Arial Narrow',sans-serif;font-size:clamp(2.6rem,6vw,5rem);letter-spacing:.04em;line-height:.93;margin:.3rem 0 .7rem;text-shadow:0 4px 30px rgba(0,0,0,.8)}
.tl{color:var(--b);font-style:italic;margin-bottom:.7rem}
.ov{color:#d8d8e6;line-height:1.65;font-size:.98rem;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
.meta{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.4rem 0 .6rem}
.chip{padding:.28rem .78rem;border-radius:999px;font-size:.74rem;font-weight:600;letter-spacing:.04em;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);backdrop-filter:blur(8px)}
.chip.star{background:rgba(255,209,102,.15);border-color:rgba(255,209,102,.55);color:var(--gold)}
.chip.dna{background:rgba(8,217,214,.1);border-color:rgba(8,217,214,.4);color:var(--b);animation:pulse 3s ease-in-out infinite}
.cast{margin-top:.9rem;color:var(--mut);font-size:.82rem}.cast b{color:var(--txt)}
.dnaRow{margin-top:.9rem;display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;font-size:.72rem;color:var(--mut);letter-spacing:.12em;text-transform:uppercase}
.btn{display:inline-block;margin:1.1rem .6rem 0 0;padding:.75rem 1.6rem;border-radius:13px;font-weight:800;text-decoration:none!important;color:#fff!important;
 background:linear-gradient(135deg,#ff2e63,#ff6b3d);box-shadow:0 10px 30px rgba(255,46,99,.5);transition:.25s;position:relative;overflow:hidden}
.btn.ghost{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.25);box-shadow:none}
.btn:hover{transform:translateY(-3px) scale(1.05);box-shadow:0 18px 44px rgba(255,46,99,.65)}
@keyframes pulse{50%{box-shadow:0 0 18px rgba(8,217,214,.55)}}

/* section titles */
.sec{display:flex;align-items:center;gap:.9rem;margin:1.8rem 0 1.1rem}
.sec h2{font-family:'Bebas Neue',Impact,'Arial Narrow',sans-serif;font-size:2.1rem;letter-spacing:.12em;margin:0;white-space:nowrap}
.sec h2 em{font-style:normal;background:linear-gradient(90deg,var(--a),var(--gold));-webkit-background-clip:text;background-clip:text;color:transparent}
.sec .line{flex:1;height:2px;background:linear-gradient(90deg,var(--a),var(--c),transparent);border-radius:2px;opacity:.7}

/* cards */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:1.4rem;perspective:1200px}
a.card{--rx:0deg;--ry:0deg;--ty:0px;--s:1;--mx:50%;--my:50%;position:relative;display:block;aspect-ratio:2/3;border-radius:18px;overflow:hidden;text-decoration:none!important;color:#fff!important;
 background:var(--card);border:1px solid rgba(255,255,255,.08);transform:perspective(800px) rotateX(var(--rx)) rotateY(var(--ry)) translateY(var(--ty)) scale(var(--s));
 transition:transform .18s ease-out,box-shadow .35s,border-color .35s;animation:rise .6s ease both;will-change:transform}
a.card .img{position:absolute;inset:-4%;background-size:cover;background-position:center;transition:transform .7s}
a.card .ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:'Bebas Neue',Impact,'Arial Narrow',sans-serif;font-size:7rem;opacity:.3}
a.card:hover{--ty:-12px;--s:1.04;box-shadow:0 30px 60px rgba(0,0,0,.75),0 0 45px rgba(255,46,99,.5);z-index:5;border-color:rgba(255,46,99,.8)}
a.card:hover .img{transform:scale(1.12)}
a.card::after{content:"";position:absolute;inset:0;pointer-events:none;opacity:0;transition:opacity .3s;background:radial-gradient(circle at var(--mx) var(--my),rgba(255,255,255,.3),transparent 45%);mix-blend-mode:overlay}
a.card:hover::after{opacity:1}
a.card .fade{position:absolute;inset:0;background:linear-gradient(0deg,rgba(4,4,9,.97) 0%,rgba(4,4,9,.6) 40%,transparent 66%)}
a.card .info{position:absolute;left:0;right:0;bottom:0;padding:.95rem;transform:translateZ(30px)}
a.card .t{font-weight:800;font-size:.96rem;line-height:1.15;margin-bottom:.3rem}
a.card .g{font-size:.68rem;color:#cacade;opacity:.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
a.card .bar{height:4px;border-radius:9px;background:rgba(255,255,255,.15);margin-top:.55rem;overflow:hidden}
a.card .bar i{display:block;height:100%;background:linear-gradient(90deg,var(--b),var(--a));box-shadow:0 0 12px var(--b)}
a.card .rate{position:absolute;top:.65rem;right:.65rem;padding:.22rem .58rem;border-radius:10px;font-size:.76rem;font-weight:800;color:var(--gold);background:rgba(0,0,0,.65);backdrop-filter:blur(6px);border:1px solid rgba(255,209,102,.45)}
a.card .rank{position:absolute;top:.4rem;left:.7rem;font-family:'Bebas Neue',Impact,'Arial Narrow',sans-serif;font-size:2.3rem;color:#fff;text-shadow:0 2px 14px #000;-webkit-text-stroke:1px rgba(255,46,99,.8)}
.sim{font-size:.66rem;color:var(--b);font-weight:700;letter-spacing:.08em;margin-top:.4rem}
.why{font-size:.64rem;color:var(--gold);opacity:.9;margin-top:.15rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* controls */
div[data-baseweb="select"]>div{background:rgba(255,255,255,.07)!important;border-radius:16px!important;border:1px solid rgba(255,255,255,.18)!important;min-height:54px;box-shadow:0 0 30px rgba(160,102,255,.15)}
.stTextInput input{border-radius:12px!important;background:rgba(255,255,255,.07)!important}
.stButton>button{width:100%;border-radius:13px;font-weight:700;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.06);color:#fff;transition:.25s}
.stButton>button:hover{background:linear-gradient(135deg,#ff2e63,#a066ff);border-color:transparent;transform:translateY(-2px);color:#fff}
button[data-baseweb="tab"]{font-weight:800;letter-spacing:.05em;font-size:1rem}
.status{font-size:.78rem;padding:.6rem .85rem;border-radius:12px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1)}
.empty{text-align:center;padding:2.5rem;color:var(--mut);border:1px dashed rgba(255,255,255,.22);border-radius:18px}
.foot{text-align:center;color:var(--mut);font-size:.72rem;letter-spacing:.25em;text-transform:uppercase;margin:3rem 0 1rem}
@keyframes rise{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}
@keyframes drift{from{transform:scale(1.04)}to{transform:scale(1.14) translateX(-1.5%)}}

/* force the dark look even if the viewer's OS/browser is in light mode */
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{color:var(--txt)}
[data-testid="stWidgetLabel"] p,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,.stMarkdown p,.stMarkdown h3,.stMarkdown h5,
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span,section[data-testid="stSidebar"] h3{color:#e9e9f3!important}
[data-testid="stCaptionContainer"] p{color:var(--mut)!important}
button[data-baseweb="tab"]{color:#c9c9dc!important;background:transparent!important}
button[data-baseweb="tab"][aria-selected="true"]{color:#fff!important}
div[data-baseweb="select"] div,div[data-baseweb="select"] input,div[data-baseweb="input"],div[data-baseweb="base-input"],.stTextInput input{background-color:rgba(255,255,255,.07)!important;color:#fff!important}
div[data-baseweb="select"] svg{fill:#fff!important}
div[data-baseweb="select"] input::placeholder,.stTextInput input::placeholder{color:#a9a9c4!important;opacity:1}
div[data-baseweb="select"] span{color:#fff!important}
div[data-baseweb="popover"] *,ul[role="listbox"],li[role="option"]{background-color:#15152a!important;color:#fff!important}
li[role="option"]:hover,li[aria-selected="true"]{background-color:#2a2150!important}
.stButton>button p{color:#fff!important}

/* version-proof widget theming (testid + ARIA selectors) */
[data-testid="stSelectbox"] div,[data-testid="stMultiSelect"] div,[data-testid="stTextInput"] div,[data-testid="stSelectbox"] input,[data-testid="stTextInput"] input,[data-testid="stMultiSelect"] input{background-color:#181830!important;color:#fff!important}
[data-testid="stSelectbox"] input::placeholder,[data-testid="stTextInput"] input::placeholder,[data-testid="stMultiSelect"] input::placeholder{color:#a9a9c4!important;opacity:1}
[data-testid="stSelectbox"] svg,[data-testid="stMultiSelect"] svg,[data-testid="stTextInput"] svg{fill:#fff!important;color:#fff!important}
[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div{border-radius:16px!important;border:1px solid rgba(255,255,255,.2)!important;box-shadow:0 0 30px rgba(160,102,255,.2)}
[data-testid="stSelectbox"] [data-baseweb],[data-testid="stSelectbox"]>div>div{min-height:54px}
[role="listbox"],[role="listbox"] *,[role="option"],[data-testid="stSelectboxVirtualDropdown"],[data-testid="stSelectboxVirtualDropdown"] *{background-color:#15152a!important;color:#fff!important}
[role="option"]:hover,[role="option"][aria-selected="true"],[role="option"]:hover *{background-color:#2c2256!important}
[role="tab"]{color:#c9c9dc!important;background:transparent!important;font-weight:800;letter-spacing:.05em}
[role="tab"][aria-selected="true"]{color:#fff!important}
.status,.status *{color:#e9e9f3!important}
#cm-glow{position:fixed;width:520px;height:520px;border-radius:50%;pointer-events:none;z-index:1;transform:translate(-50%,-50%);
 background:radial-gradient(circle,rgba(160,102,255,.2),rgba(255,46,99,.08) 40%,transparent 68%);transition:opacity .3s}
#cm-dust{position:fixed;inset:0;z-index:1;pointer-events:none}

/* ── v2 upgrades: tiles, match ring, compare, taste bars ── */
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin:0 0 1.2rem}
.tile{padding:.9rem 1.1rem;border-radius:16px;background:linear-gradient(135deg,rgba(255,255,255,.09),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.1);
  backdrop-filter:blur(8px);transition:.3s;animation:rise .6s ease both}
.tile:hover{transform:translateY(-5px);border-color:rgba(8,217,214,.55);box-shadow:0 16px 36px rgba(8,217,214,.16)}
.tile b{display:block;font-family:'Bebas Neue',Impact,'Arial Narrow',sans-serif;font-size:1.9rem;letter-spacing:.05em;color:var(--gold);line-height:1.1}
.tile span{color:var(--mut);font-size:.66rem;letter-spacing:.18em;text-transform:uppercase}
@media(max-width:800px){.tiles{grid-template-columns:repeat(2,1fr)}}
.stat{transition:.3s;position:relative;overflow:hidden}
.stat::before{content:"";position:absolute;left:0;top:0;height:3px;width:100%;background:linear-gradient(90deg,var(--a),var(--gold),var(--b));background-size:200% 100%;animation:flow 5s linear infinite}
.stat:hover{transform:translateY(-5px);box-shadow:0 18px 40px rgba(160,102,255,.22)}

.ring{position:absolute;right:.8rem;bottom:.85rem;width:46px;height:46px;border-radius:50%;display:grid;place-items:center;
  background:conic-gradient(var(--b) calc(var(--p)*1%),rgba(255,255,255,.16) 0);box-shadow:0 0 16px rgba(8,217,214,.35)}
.ring::before{content:"";position:absolute;inset:4px;border-radius:50%;background:#0a0a15}
.ring b{position:relative;font-size:.64rem;font-weight:800;color:#fff}
.ring.big{position:relative;right:auto;bottom:auto;width:132px;height:132px;margin:0 auto}
.ring.big::before{inset:9px}.ring.big b{font-size:1.7rem;font-family:'Bebas Neue',Impact,sans-serif;letter-spacing:.04em}
a.card .info.m .t,a.card .info.m .g,a.card .info.m .why{padding-right:3.1rem}

.vs{display:grid;grid-template-columns:1fr 190px 1fr;gap:1.2rem;align-items:center;padding:1.6rem;border-radius:24px;margin:.6rem 0 1rem;
  background:linear-gradient(135deg,rgba(255,255,255,.08),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.12);box-shadow:0 30px 70px rgba(0,0,0,.5);animation:rise .6s ease both}
.vs .side h3{font-family:'Bebas Neue',Impact,sans-serif;font-size:2rem;letter-spacing:.06em;margin:.2rem 0 .5rem;line-height:1}
.vs .mid{text-align:center}.vs .verdict{margin-top:.7rem;font-weight:800;letter-spacing:.1em;font-size:.78rem;text-transform:uppercase;color:var(--b)}
.hbar{height:8px;border-radius:9px;background:rgba(255,255,255,.12);overflow:hidden;margin:.35rem 0 .7rem}
.hbar i{display:block;height:100%;background:linear-gradient(90deg,var(--b),var(--a));border-radius:9px;box-shadow:0 0 12px var(--b)}
.lbl{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:var(--mut)}
.shared{margin:.4rem 0 1.4rem;padding:1rem 1.2rem;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);display:flex;flex-wrap:wrap;gap:.45rem;align-items:center}
@media(max-width:900px){.vs{grid-template-columns:1fr}}

.bars{display:flex;flex-direction:column;gap:.7rem;margin:.6rem 0 1.4rem}
.bars .row{display:grid;grid-template-columns:140px 1fr 46px;align-items:center;gap:.9rem;font-weight:700;font-size:.88rem}
.bars .track{height:14px;border-radius:9px;background:rgba(255,255,255,.1);overflow:hidden}
.bars .track i{display:block;height:100%;border-radius:9px;background:linear-gradient(90deg,var(--a),var(--c),var(--b));box-shadow:0 0 14px rgba(160,102,255,.6);animation:grow .9s cubic-bezier(.2,.8,.2,1) both}
.bars .n{color:var(--gold);text-align:right}
@keyframes grow{from{width:0}}
</style>
<div class="aurora"><i></i><i></i><i></i></div><div class="grain"></div>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Browser-side magic: floating particles, cursor spotlight, mouse-tracked 3D card tilt.
# Injected into the parent page; if the browser blocks it, the app just runs without the extras.
FX_JS = (
    """
<script>
(function(){try{
 const P=window.parent, D=P.document; if(P.__cinematch) return; P.__cinematch=true;
 const glow=D.createElement('div'); glow.id='cm-glow'; D.body.appendChild(glow);
 const cv=D.createElement('canvas'); cv.id='cm-dust'; D.body.appendChild(cv);
 const ctx=cv.getContext('2d'); let W,H,pts=[];
 function size(){W=cv.width=P.innerWidth;H=cv.height=P.innerHeight}
 size(); P.addEventListener('resize',size);
 for(let i=0;i<70;i++)pts.push({x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.8+.3,v:Math.random()*.35+.08,a:Math.random()*.5+.15,h:Math.random()<.5?340:180});
 (function loop(){ctx.clearRect(0,0,W,H);
  for(const p of pts){p.y-=p.v;p.x+=Math.sin(p.y/60)*.25;if(p.y<-5){p.y=H+5;p.x=Math.random()*W}
   ctx.beginPath();ctx.fillStyle='hsla('+p.h+',100%,70%,'+p.a+')';ctx.arc(p.x,p.y,p.r,0,6.283);ctx.fill();}
  P.requestAnimationFrame(loop)})();
 D.addEventListener('mousemove',e=>{glow.style.left=e.clientX+'px';glow.style.top=e.clientY+'px';
  const c=e.target.closest&&e.target.closest('a.card'); if(!c)return;
  const r=c.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;
  c.style.setProperty('--ry',((x-.5)*18)+'deg');c.style.setProperty('--rx',((.5-y)*18)+'deg');
  c.style.setProperty('--mx',(x*100)+'%');c.style.setProperty('--my',(y*100)+'%');});
 D.addEventListener('mouseout',e=>{const c=e.target.closest&&e.target.closest('a.card');
  if(c&&!c.contains(e.relatedTarget)){c.style.setProperty('--rx','0deg');c.style.setProperty('--ry','0deg');}});
}catch(e){}})();
</script>"""
)
try:
    st.iframe(FX_JS, height=1)  # Streamlit >= 1.5x
except Exception:
    components.html(FX_JS, height=0)

# ═════════════════════════════ UI HELPERS ═════════════════════════════
def split_genres(s):
    found = [g for g in GENRES if g in (s or "")]
    return found or (s or "").split()[:3]


def gradient_for(title):
    h = int(hashlib.md5(title.encode()).hexdigest(), 16)
    return f"linear-gradient(145deg,hsl({h % 360},72%,34%),hsl({(h // 360) % 360},78%,15%))"


def for_you(n=18):
    """Personalised picks: average the TF-IDF vectors of your recent history, then find the closest films."""
    seen = [t for t, _ in History_dbHistory_db.titles_viewed(10) if t in INDICES]
    idxs = [_idx(t) for t in seen]
    if not idxs:
        return []
    profile = np.asarray(MATRIX[idxs].mean(axis=0))
    sims = cosine_similarity(profile, MATRIX).ravel()
    sims[idxs] = -1
    top = [i for i in np.argsort(sims)[::-1][:n] if sims[i] > 0]
    out = DF.iloc[top].copy()
    out["score"] = sims[top]
    return out.to_dict("records")


def taste_profile():
    """Genre counts across everything you have opened (weighted by how often)."""
    counts = {}
    for t, times in history_db.titles_viewed():
        if t in INDICES:
            for g in split_genres(get_movie(t)["genres"]):
                counts[g] = counts.get(g, 0) + times
    return sorted(counts.items(), key=lambda kv: -kv[1])


def tiles_html(m):
    pop = max(1, 100 - round((DF["popularity"] < m["popularity"]).mean() * 100))
    rank = int((DF["vote_average"] > m["vote_average"]).sum()) + 1
    items = [(f'{m["vote_average"]:.1f}<small style="font-size:1rem;color:#9a9ab0"> /10</small>', "Audience rating"),
             (f"#{rank:,}", f"Rating rank of {len(DF):,}"), (f"Top {pop}%", "Popularity"),
             (str(len(split_genres(m["genres"]))), "Genres blended")]
    return '<div class="tiles">' + "".join(f'<div class="tile"><b>{v}</b><span>{k}</span></div>' for v, k in items) + "</div>"


def compare_html(a, b):
    ma, mb = get_movie(a), get_movie(b)
    ia, ib = _idx(a), _idx(b)
    pct = round(float(cosine_similarity(MATRIX[ia], MATRIX[ib])[0, 0]) * 100)
    verdict = "Very similar" if pct >= 30 else "Some overlap" if pct >= 12 else "Quite different"
    ga, gb = split_genres(ma["genres"]), split_genres(mb["genres"])
    common = [g for g in ga if g in gb]

    def side(m, gs):
        chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in gs)
        pop = max(1, 100 - round((DF["popularity"] < m["popularity"]).mean() * 100))
        return (f'<div class="side"><h3>{html.escape(m["title"])}</h3><div class="meta">{chips}</div>'
                f'<div class="lbl">Rating · {m["vote_average"]:.1f} / 10</div><div class="hbar"><i style="width:{m["vote_average"] * 10}%"></i></div>'
                f'<div class="lbl">Popularity · top {pop}%</div><div class="hbar"><i style="width:{101 - pop}%"></i></div></div>')

    shared = "".join(f'<span class="chip dna">{html.escape(w)}</span>' for w in shared_terms(ia, ib, 8))
    same = "".join(f'<span class="chip star">{html.escape(g)}</span>' for g in common)
    return (f'<div class="vs">{side(ma, ga)}<div class="mid"><div class="ring big" style="--p:{min(100, pct * 2)}"><b>{pct}%</b></div>'
            f'<div class="verdict">{verdict}</div></div>{side(mb, gb)}</div>'
            f'<div class="shared"><span class="lbl">🧬 Shared DNA</span>{shared or "<span class=lbl>no shared keywords</span>"}</div>'
            f'<div class="shared"><span class="lbl">🎭 Genres in common</span>{same or "<span class=lbl>none</span>"}</div>')


def taste_html(profile):
    top = profile[0][1]
    rows = "".join(f'<div class="row"><span>{html.escape(g)}</span><div class="track"><i style="width:{round(c / top * 100)}%"></i></div><span class="n">{c}</span></div>'
                   for g, c in profile[:8])
    return f'<div class="bars">{rows}</div>'


def card_html(m, info, rank=None, show_sim=True):
    t = html.escape(m["title"])
    poster = (info or {}).get("poster")
    bg = (f'<div class="img" style="background-image:url({poster})"></div>' if poster else
          f'<div class="img" style="background:{gradient_for(m["title"])}"></div><div class="ph">{t[:1]}</div>')
    genres = html.escape(" · ".join(split_genres(m["genres"])[:3]))
    sim = why = ""
    if show_sim and "score" in m:
        pct = max(1, min(100, round(m["score"] * 100)))
        sim = f'<div class="ring" style="--p:{min(100, pct * 2)}"><b>{pct}%</b></div>'
        if m.get("shared"):
            why = f'<div class="why">🧬 {html.escape(", ".join(m["shared"][:2]))}</div>'
    tip = html.escape(f'{m["title"]} — ★{m["vote_average"]:.1f}', quote=True)
    rk = f'<div class="rank">{rank}</div>' if rank else ""
    delay = f"animation-delay:{(rank or 1) * 0.05:.2f}s"
    return (f'<a class="card" title="{tip}" target="_self" style="{delay}" href="?movie={quote(m["title"])}">{bg}<div class="fade"></div>{rk}'
            f'<div class="rate">★ {m["vote_average"]:.1f}</div><div class="info{" m" if sim else ""}"><div class="t">{t}</div>'
            f'<div class="g">{genres}</div>{sim}{why}</div></a>')


def grid(movies, numbered=False, show_sim=True):
    infos = tmdb_many(movies)
    cards = "".join(card_html(m, infos.get(int(m["tmdb_id"])), rank=i if numbered else None, show_sim=show_sim)
                    for i, m in enumerate(movies, 1))
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)


def section(title, accent=""):
    st.markdown(f'<div class="sec"><h2>{html.escape(title)} <em>{html.escape(accent)}</em></h2><div class="line"></div></div>',
                unsafe_allow_html=True)


def hero(m):
    info = tmdb_many([m]).get(int(m["tmdb_id"])) or {}
    t = html.escape(m["title"])
    bg = info.get("backdrop") or info.get("poster")
    bgcss = f"background-image:url({bg})" if bg else f"background:{gradient_for(m['title'])}"
    chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in split_genres(m["genres"]))
    extra = (f'<span class="chip">{info["year"]}</span>' if info.get("year") else "") + \
            (f'<span class="chip">{info["runtime"]} min</span>' if info.get("runtime") else "")
    cast = f'<div class="cast"><b>Starring</b> &nbsp;{html.escape(" · ".join(info["cast"]))}</div>' if info.get("cast") else ""
    yt = info.get("trailer") or f"https://www.youtube.com/results?search_query={quote(m['title'] + ' official trailer')}"
    tagline = (info.get("tagline") or m.get("tagline") or "").strip()
    tl = f'<div class="tl">“{html.escape(tagline)}”</div>' if tagline else ""
    words = "".join(f'<span class="chip dna">{html.escape(w)}</span>' for w in dna(m["title"]))
    dna_row = f'<div class="dnaRow">🧬 Movie DNA &nbsp;{words}</div>' if words else ""
    st.markdown(
        f'<div class="hero"><div class="bg" style="{bgcss}"></div><div class="shade"></div><div class="scan"></div><div class="body">'
        f'<div class="meta"><span class="chip star">★ {m["vote_average"]:.1f} / 10</span>{extra}{chips}</div>'
        f'<h1>{t}</h1>{tl}<div class="ov">{html.escape(m["overview"])}</div>{cast}{dna_row}'
        f'<a class="btn" target="_blank" href="{yt}">▶ &nbsp;Watch trailer</a>'
        f'<a class="btn ghost" target="_self" href="?">✕ &nbsp;Close</a></div></div>', unsafe_allow_html=True)


def ticker():
    items = "".join(f'<b>{html.escape(m["title"]).upper()}</b> ★{m["vote_average"]:.1f}<span>✦</span>' for m in trending(24))
    st.markdown(f'<div class="ticker"><div>{items}{items}</div></div>', unsafe_allow_html=True)


# ═════════════════════════════ STATE ═════════════════════════════
qp = st.query_params
current = qp.get("movie") if qp.get("movie") in INDICES else None
mood = qp.get("mood") if qp.get("mood") in MOODS else None

# ═════════════════════════════ SIDEBAR ═════════════════════════════
with st.sidebar:
    st.markdown("### 🎛️ Tune your taste")
    n = st.slider("How many picks", 6, 30, 12, step=2)
    min_rating = st.slider("Only show picks rated at least ★", 0.0, 9.0, 0.0, step=0.5, format="%.1f",
                           help="Filters the recommendations by their TMDB score (out of 10). "
                                "It is not a rating for the selected movie. 0 = show everything.")
    st.caption("Showing all ratings" if min_rating == 0 else f"Hiding picks rated below ★ {min_rating:.1f} / 10")
    genres = st.multiselect("Must include genres", GENRES)
    sort_by = st.selectbox("Sort picks by", ["Best match", "Highest rated", "Most popular"])
    c1, c2 = st.columns(2)
    if c1.button("🎲 Surprise"):
        st.query_params.clear()
        st.query_params["movie"] = DF[DF["vote_average"] >= 6.5].sample(1)["title"].iloc[0]
        st.session_state["next_source"] = "surprise"
        st.rerun()
    if c2.button("🏠 Home"):
        st.query_params.clear()
        st.rerun()
    st.markdown("---")

    # ── History (SQLite) ──
    # Every search / opened movie is stored once (not on every slider rerun).
    if current and st.session_state.get("last_logged") != current:
        history_db.add_view(current, source=st.session_state.pop("next_source", "click"), n_picks=n, min_rating=min_rating, genres=genres)
        st.session_state["last_logged"] = current
    if not current:
        st.session_state["last_logged"] = None

    st.markdown("### 📜 History")
    recent_items = history_db.recent(8)
    if recent_items:
        rows_html = []
        for r in recent_items:
            badge = f' <span style="opacity:.6">×{r["times"]}</span>' if r["times"] > 1 else ""
            rows_html.append(f'<div style="margin:4px 0"><a target="_self" href="?movie={quote(r["title"])}">'
                             f'{html.escape(r["title"])}</a>{badge}</div>')
        links = "".join(rows_html)
        st.markdown(links, unsafe_allow_html=True)
        top = history_db.most_viewed(1)
        st.caption(f"❤️ {len(history_db.wl_list())} saved · {history_db.count()} views logged · most viewed: {top[0]['title']}")
        h1, h2 = st.columns(2)
        h1.download_button("⬇ CSV", pd.DataFrame(history_db.all_rows()).to_csv(index=False),
                           file_name="cinematch_history.csv", mime="text/csv")
        if h2.button("🗑 Clear"):
            history_db.clear()
            st.session_state["last_logged"] = None
            st.rerun()
    else:
        st.caption("Nothing yet. Open a movie and it will appear here.")
    st.markdown("---")
    st.caption("Model: TF-IDF (1–2 grams, 5000 features) + cosine similarity.")

# ═════════════════════════════ HEADER ═════════════════════════════
ticker()
st.markdown('<div class="logo" data-t="CINEMATCH">CINEMATCH</div>'
            '<div class="tag">One movie you love &nbsp;→&nbsp; <b>your next obsession</b></div>', unsafe_allow_html=True)

choice = st.selectbox("Search", TITLES, index=TITLES.index(current) if current else None,
                      placeholder="🔍  Type a movie you love…  (Inception, Titanic, Avatar)", label_visibility="collapsed")
if choice and choice != current:
    st.query_params.clear()
    st.query_params["movie"] = choice
    st.session_state["next_source"] = "search"
    st.rerun()

# ═════════════════════════════ PAGES ═════════════════════════════
if current:
    movie = get_movie(current)
    hero(movie)
    st.markdown(tiles_html(movie), unsafe_allow_html=True)
    if st.session_state.get("flash"):
        st.toast(st.session_state.pop("flash"))
    saved = history_db.wl_has(current)
    w1, _w2 = st.columns([1, 3])
    if w1.button("💔 Remove from watchlist" if saved else "❤️ Save to watchlist", key="wl_btn"):
        (history_db.wl_remove if saved else history_db.wl_add)(current)
        st.session_state["flash"] = "Removed from watchlist" if saved else "Saved to your watchlist ❤️"
        st.rerun()

    recs = recommend(current, n=n, min_rating=min_rating, genres=genres)
    if sort_by == "Highest rated":
        recs.sort(key=lambda r: -r["vote_average"])
    elif sort_by == "Most popular":
        recs.sort(key=lambda r: -r["popularity"])

    tab_r, tab_c = st.tabs(["🎯 Recommendations", "⚔️ Compare"])
    with tab_r:
        section("Because you liked", current)
        if recs:
            grid(recs, numbered=True)
            st.caption("💡 Hover for 3D tilt · the ring shows how closely each film matches · 🧬 shows the words that link it to your pick.")
        else:
            st.markdown('<div class="empty">😕 No matches with these filters. Lower the minimum rating or remove a genre.</div>',
                        unsafe_allow_html=True)
    with tab_c:
        section("Head to", "head")
        other = st.selectbox("Compare with", [t for t in TITLES if t != current], index=None, key="cmp",
                             placeholder="🔍  Pick another movie to compare…", label_visibility="collapsed")
        if other:
            st.markdown(compare_html(current, other), unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">⚔️ Choose a second movie to see how similar the two really are.</div>', unsafe_allow_html=True)

elif mood:
    section(f"{MOODS[mood]} {mood}", "portal")
    grid(pick_genre(mood, 24), show_sim=False)

else:
    st.markdown(f'<div class="stats"><div class="stat"><strong>{len(DF):,}</strong><span>Films indexed</span></div>'
                f'<div class="stat"><strong>{MATRIX.shape[1]:,}</strong><span>TF-IDF features</span></div>'
                f'<div class="stat"><strong>{len(GENRES)}</strong><span>Genres</span></div>'
                f'<div class="stat"><strong>{history_db.count()}</strong><span>Your views</span></div></div>', unsafe_allow_html=True)
    past = history_db.recent(12)
    if past:
        section("Your", "History")
        st.markdown('<div class="moods">' + "".join(
            f'<a class="mood" target="_self" href="?movie={quote(r["title"])}"><span class="e">🕘</span>{html.escape(r["title"])}</a>'
            for r in past) + "</div>", unsafe_allow_html=True)
    section("Pick a", "mood")
    st.markdown('<div class="moods">' + "".join(
        f'<a class="mood" target="_self" href="?mood={quote(g)}"><span class="e">{e}</span>{g}</a>' for g, e in MOODS.items()) + "</div>",
        unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6 = st.tabs(["🔥 Trending", "🏆 Top rated", "💎 Hidden gems", "✨ For you", "❤️ Watchlist", "📊 Your taste"])
    with t1:
        grid(trending(18), show_sim=False)
    with t2:
        grid(top_rated(18), show_sim=False)
    with t3:
        st.caption("Highly rated films that flew under the radar.")
        grid(hidden_gems(18), show_sim=False)
    with t4:
        picks = for_you(18)
        if picks:
            st.caption("Built from the average taste of the movies in your History.")
            grid(picks)
        else:
            st.markdown('<div class="empty">✨ Open a few movies and personalised picks will appear here.</div>', unsafe_allow_html=True)
    with t5:
        saved_movies = [get_movie(t) for t in history_db.wl_list() if t in INDICES]
        if saved_movies:
            grid(saved_movies, show_sim=False)
        else:
            st.markdown('<div class="empty">❤️ Your watchlist is empty. Open a movie and tap “Save to watchlist”.</div>', unsafe_allow_html=True)
    with t6:
        prof = taste_profile()
        if prof:
            st.markdown(f'<div class="tiles"><div class="tile"><b>{prof[0][0]}</b><span>Favourite genre</span></div>'
                        f'<div class="tile"><b>{len(history_db.titles_viewed())}</b><span>Movies explored</span></div>'
                        f'<div class="tile"><b>{history_db.count()}</b><span>Total views</span></div>'
                        f'<div class="tile"><b>{len(history_db.wl_list())}</b><span>Saved</span></div></div>', unsafe_allow_html=True)
            section("Your genre", "DNA")
            st.markdown(taste_html(prof), unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">📊 Your taste profile builds as you explore movies.</div>', unsafe_allow_html=True)

st.markdown('<div class="foot">CineMatch · TF-IDF × cosine similarity · posters by TMDB</div>', unsafe_allow_html=True)