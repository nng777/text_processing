import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime
import statistics

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib may not be installed
    plt = None

# Third party imports (nltk, spacy, langdetect) are optional at runtime
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - nltk may not be installed
    nltk = None

try:
    import spacy
except Exception:  # pragma: no cover - spacy may not be installed
    spacy = None

try:
    from langdetect import detect
except Exception:  # pragma: no cover - langdetect may not be installed
    detect = None

# Sample posts provided in ``extraction.py``
# Extended with simple engagement metrics and timestamps for the
# bonus analysis tasks.
sample_posts = [
    {
        "user": "1st_user",
        "timestamp": "2025-07-10 08:05",
        "text": "Just launched our new #AI product! 🚀 Thanks to @team_awesome for the hard work. Check it out: https://oursite.com/product #innovation #tech",
        "likes": 150,
        "shares": 20,
    },
    {
        "user": "2nd_user",
        "timestamp": "2025-07-10 09:30",
        "text": "Beautiful sunset in San Francisco today 🌅 #photography #california #nature Love this city! @visit_sf",
        "likes": 85,
        "shares": 12,
    },
    {
        "user": "3rd_user",
        "timestamp": "2025-07-11 14:10",
        "text": "Breaking: Apple announces new iPhone with 50% better battery life! Stock up 12% 📈 #Apple #iPhone #tech #stocks",
        "likes": 320,
        "shares": 50,
    },
    {
        "user": "4th_user",
        "timestamp": "2025-07-12 18:45",
        "text": "Feeling grateful for 10,000 followers! 🙏 You all are amazing. Special thanks to @mentor_john for the guidance #milestone #grateful",
        "likes": 60,
        "shares": 8,
    },
]


@dataclass
class RegExResult:
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    emojis: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)


@dataclass
class NLTKResult:
    sentences: int = 0
    words: int = 0
    sentiment: Dict[str, float] = field(default_factory=dict)
    keywords: List[Tuple[str, int]] = field(default_factory=list)
    readability: float = 0.0


@dataclass
class SpaCyResult:
    entities: List[Tuple[str, str]] = field(default_factory=list)
    language: str = ""
    dependencies: List[Tuple[str, str, str]] = field(default_factory=list)


class SocialMediaAnalyzer:
    """Analyzer implementing RegEx, NLTK and SpaCy processing."""

    # RegEx patterns. Explanations provided inline.
    PATTERNS = {
        # Matches words starting with '#', capturing typical hashtags
        'hashtags': r'(?<!\w)#\w+',
        # Matches words starting with '@' for mentions
        'mentions': r'(?<!\w)@\w+',
        # Matches http or https URLs
        'urls': r'https?://[^\s]+',
        # Basic emoji pattern covering common emoji ranges
        'emojis': r'[\U0001F300-\U0001FAFF]',
        # Numbers including comma separators (follower counts, etc.)
        'numbers': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
    }

    def __init__(self) -> None:
        # Prepare NLTK tools if available
        if nltk:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            except Exception:
                pass
            self.stop_words = set(stopwords.words('english'))
            self.sentiment = SentimentIntensityAnalyzer()
        else:
            self.stop_words = set()
            self.sentiment = None

        # Load SpaCy model if available
        if spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except Exception:
                self.nlp = None
        else:
            self.nlp = None

    # --------------------------- RegEx Phase ---------------------------
    def extract_with_regex(self, text: str) -> RegExResult:
        res = RegExResult()
        for key, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            setattr(res, key, matches)
        return res

    # --------------------------- NLTK Phase ---------------------------
    def analyze_with_nltk(self, text: str) -> NLTKResult:
        result = NLTKResult()
        if not nltk:
            return result

        sentences = sent_tokenize(text)
        tokens = word_tokenize(text)
        words = [t.lower() for t in tokens if t.isalpha()]
        words_filtered = [w for w in words if w not in self.stop_words]

        # Sentiment
        scores = self.sentiment.polarity_scores(text)

        # Keyword extraction via most common filtered words
        freq = Counter(words_filtered)
        common = freq.most_common(5)

        # Simple readability metric: average word length
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0

        result.sentences = len(sentences)
        result.words = len(words)
        result.sentiment = scores
        result.keywords = common
        result.readability = avg_word_len
        return result

    # --------------------------- SpaCy Phase ---------------------------
    def analyze_with_spacy(self, text: str) -> SpaCyResult:
        result = SpaCyResult()
        if not self.nlp:
            return result

        doc = self.nlp(text)
        result.entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Language detection via langdetect if available
        if detect:
            try:
                result.language = detect(text)
            except Exception:
                result.language = "unknown"

        # Extract simple subject-verb-object triples
        dependencies = []
        for token in doc:
            if token.dep_ == 'ROOT':
                subject = [w.text for w in token.lefts if w.dep_ in {'nsubj', 'nsubjpass'}]
                obj = [w.text for w in token.rights if w.dep_ in {'dobj', 'pobj'}]
                if subject and obj:
                    dependencies.append((subject[0], token.text, obj[0]))
        result.dependencies = dependencies
        return result

    # --------------------------- Combined ---------------------------
    def analyze_post(self, post: Dict[str, object]) -> Dict[str, object]:
        """Analyze a single post dictionary."""
        text = post["text"]
        return {
            "regex": self.extract_with_regex(text),
            "nltk": self.analyze_with_nltk(text),
            "spacy": self.analyze_with_spacy(text),
        }

    # --------------------------- Spam Detection ---------------------------
    def is_spam(self, post: Dict[str, object], analysis: Dict[str, object]) -> bool:
        """Heuristic spam/bot detection."""
        regex: RegExResult = analysis["regex"]
        nltk_res: NLTKResult = analysis["nltk"]

        hashtag_ratio = len(regex.hashtags) / nltk_res.words if nltk_res.words else 0
        suspicious_words = {"buy", "free", "click", "subscribe", "follow"}
        text_lower = post["text"].lower()
        suspicious = any(w in text_lower for w in suspicious_words)

        return (
            hashtag_ratio > 0.3
            or suspicious
            or len(regex.mentions) > 5
            or len(regex.hashtags) > 5
            or (nltk is not None and nltk_res.words == 0)
        )

    # --------------------------- Trending Topics ---------------------------
    @staticmethod
    def trending_topics(analyses: List[Dict[str, object]], top_n: int = 3) -> List[Tuple[str, int]]:
        """Return top hashtags across all posts."""
        counter: Counter = Counter()
        for data in analyses:
            regex: RegExResult = data["regex"]
            counter.update(map(str.lower, regex.hashtags))
        return counter.most_common(top_n)

    # --------------------------- Engagement Correlation ---------------------------
    @staticmethod
    def engagement_correlation(posts: List[Dict[str, object]]) -> float | None:
        """Correlation between posting hour and engagement."""
        hours: List[int] = []
        engagement: List[int] = []
        for p in posts:
            try:
                dt = datetime.strptime(p["timestamp"], "%Y-%m-%d %H:%M")
            except Exception:
                continue
            hours.append(dt.hour)
            engagement.append(p.get("likes", 0) + p.get("shares", 0))
        if len(hours) < 2:
            return None
        return statistics.correlation(hours, engagement)

    # --------------------------- Visualizations ---------------------------
    @staticmethod
    def create_visualizations(posts: List[Dict[str, object]], analyses: List[Dict[str, object]]) -> None:
        if not plt:
            return

        # Trending hashtag bar chart
        trend = SocialMediaAnalyzer.trending_topics(analyses, top_n=5)
        if trend:
            labels, counts = zip(*trend)
            plt.figure(figsize=(6, 4))
            plt.bar(labels, counts, color="skyblue")
            plt.title("Top Hashtags")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("trending_hashtags.png")

        # Engagement correlation scatter
        hours = []
        engagement = []
        for p in posts:
            try:
                dt = datetime.strptime(p["timestamp"], "%Y-%m-%d %H:%M")
            except Exception:
                continue
            hours.append(dt.hour)
            engagement.append(p.get("likes", 0) + p.get("shares", 0))
        if hours:
            plt.figure(figsize=(6, 4))
            plt.scatter(hours, engagement, color="green")
            plt.xlabel("Hour of Day")
            plt.ylabel("Engagement")
            plt.title("Engagement vs Posting Time")
            plt.tight_layout()
            plt.savefig("engagement_correlation.png")


def main(posts: List[Dict[str, object]]) -> None:
    analyzer = SocialMediaAnalyzer()
    analyses = [analyzer.analyze_post(p) for p in posts]

    # Write simple report
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('# Social Media Analysis Report\n\n')
        for i, (post, data) in enumerate(zip(posts, analyses), 1):
            f.write(f'## Post {i} by {post["user"]}\n')
            f.write(post["text"] + '\n')
            f.write(f'*Likes:* {post["likes"]}  *Shares:* {post["shares"]}\n\n')
            regex = data['regex']
            f.write('- RegEx:\n')
            f.write(f'  - Hashtags: {regex.hashtags}\n')
            f.write(f'  - Mentions: {regex.mentions}\n')
            f.write(f'  - URLs: {regex.urls}\n')
            f.write(f'  - Emojis: {regex.emojis}\n')
            f.write(f'  - Numbers: {regex.numbers}\n')

            f.write('- NLTK:\n')
            nltk_res: NLTKResult = data['nltk']
            if nltk and nltk_res.words:
                f.write(f'  - Sentiment: {nltk_res.sentiment}\n')
                kw = ', '.join(f'{w}({c})' for w, c in nltk_res.keywords)
                f.write(f'  - Keywords: {kw}\n')
                f.write(f'  - Avg word length: {nltk_res.readability:.2f}\n')
            else:
                f.write('  - NLTK not available\n')

            f.write('- SpaCy:\n')
            spacy_res: SpaCyResult = data['spacy']
            if spacy and (spacy_res.entities or spacy_res.language or spacy_res.dependencies):
                if spacy_res.entities:
                    ents = ', '.join(f'{t}({l})' for t, l in spacy_res.entities)
                    f.write(f'  - Entities: {ents}\n')
                if spacy_res.language:
                    f.write(f'  - Language: {spacy_res.language}\n')
                if spacy_res.dependencies:
                    dep = '; '.join(f'{s}-{v}-{o}' for s, v, o in spacy_res.dependencies)
                    f.write(f'  - Dependencies: {dep}\n')
            else:
                f.write('  - SpaCy not available\n')

            if analyzer.is_spam(post, data):
                f.write('- Potential spam/bot account\n')
            f.write('\n')

        trend = SocialMediaAnalyzer.trending_topics(analyses)
        if trend:
            f.write('## Trending Hashtags\n')
            for tag, count in trend:
                f.write(f'- {tag}: {count}\n')
            f.write('\n')

        corr = SocialMediaAnalyzer.engagement_correlation(posts)
        if corr is not None:
            f.write(f'**Engagement correlation with posting hour:** {corr:.2f}\n\n')

    SocialMediaAnalyzer.create_visualizations(posts, analyses)
    print('Analysis written to analysis_report.md')


if __name__ == '__main__':
    main(sample_posts)