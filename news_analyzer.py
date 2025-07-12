#!/usr/bin/env python3
"""
News Article Analyzer
A practical project demonstrating text processing with RegEx, NLTK, and SpaCy

This script analyzes news articles to extract:
- Key information (dates, emails, URLs)
- Named entities (people, organizations, locations)
- Sentiment analysis
- Text statistics and readability
"""

import re
import nltk
import spacy
from collections import Counter
from datetime import datetime

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    print("Note: Some NLTK downloads may have failed. Install manually if needed.")

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

class NewsAnalyzer:
    def __init__(self):
        print("🔧 Initializing News Analyzer...")
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize SpaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ SpaCy model loaded successfully")
        except OSError:
            print("⚠️  SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # RegEx patterns for information extraction
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'url': r'https?://[^\s]+',
            'date': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            'money': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%'
        }
        
        print("✅ Analyzer initialized successfully\n")
    
    def extract_with_regex(self, text):
        """Extract structured information using Regular Expressions"""
        print("📋 REGEX ANALYSIS: Extracting structured information...")
        
        extracted = {}
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted[pattern_name] = matches
                print(f"  • Found {len(matches)} {pattern_name}(s): {matches[:3]}{'...' if len(matches) > 3 else ''}")
        
        if not extracted:
            print("  • No structured information found")
        
        return extracted
    
    def analyze_with_nltk(self, text):
        """Analyze text using NLTK for basic NLP tasks"""
        print("\n🔤 NLTK ANALYSIS: Processing text structure...")
        
        # Tokenization
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        print(f"  • Sentences: {len(sentences)}")
        print(f"  • Words: {len(words)}")
        
        # Remove punctuation and convert to lowercase
        words_clean = [w.lower() for w in words if w.isalpha()]
        
        # Remove stop words
        words_filtered = [w for w in words_clean if w not in self.stop_words]
        
        print(f"  • Words after cleaning: {len(words_filtered)}")
        
        # Stemming
        words_stemmed = [self.stemmer.stem(w) for w in words_filtered]
        
        # Most common words
        word_freq = Counter(words_stemmed)
        common_words = word_freq.most_common(5)
        print(f"  • Most common words: {[f'{word}({count})' for word, count in common_words]}")
        
        # Part-of-speech tagging
        pos_tags = nltk.pos_tag(words[:20])  # First 20 words for brevity
        print(f"  • Sample POS tags: {pos_tags[:5]}")
        
        # Sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        sentiment = max(sentiment_scores, key=sentiment_scores.get)
        print(f"  • Sentiment: {sentiment.upper()} (score: {sentiment_scores[sentiment]:.2f})")
        
        return {
            'sentences': len(sentences),
            'words': len(words),
            'words_clean': len(words_filtered),
            'common_words': common_words,
            'sentiment': sentiment_scores,
            'pos_sample': pos_tags[:5]
        }
    
    def analyze_with_spacy(self, text):
        """Analyze text using SpaCy for advanced NLP"""
        if not self.nlp:
            print("\n⚠️  SpaCy not available - skipping advanced analysis")
            return {}
        
        print("\n🧠 SPACY ANALYSIS: Advanced NLP processing...")
        
        doc = self.nlp(text)
        
        # Named Entity Recognition
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entity_types = Counter([ent[1] for ent in entities])
        
        print(f"  • Named entities found: {len(entities)}")
        for entity_type, count in entity_types.most_common(5):
            print(f"    - {entity_type}: {count}")
        
        # Show some examples
        if entities:
            print(f"  • Entity examples: {entities[:3]}")
        
        # Lemmatization (more accurate than stemming)
        lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        lemma_freq = Counter(lemmas)
        common_lemmas = lemma_freq.most_common(5)
        print(f"  • Most common lemmas: {[f'{lemma}({count})' for lemma, count in common_lemmas]}")
        
        # Dependency parsing (show relationships)
        print("  • Sample dependencies:")
        for token in doc[:10]:  # First 10 tokens
            if token.dep_ != 'punct':
                print(f"    - {token.text} --{token.dep_}--> {token.head.text}")
        
        return {
            'entities': entities,
            'entity_types': dict(entity_types),
            'common_lemmas': common_lemmas,
            'total_tokens': len(doc)
        }
    
    def analyze_article(self, article_text, title="News Article"):
        """Complete analysis of a news article"""
        print(f"\n{'='*60}")
        print(f"📰 ANALYZING: {title}")
        print(f"{'='*60}")
        print(f"Article length: {len(article_text)} characters")
        
        # Step 1: RegEx extraction
        regex_results = self.extract_with_regex(article_text)
        
        # Step 2: NLTK analysis
        nltk_results = self.analyze_with_nltk(article_text)
        
        # Step 3: SpaCy analysis
        spacy_results = self.analyze_with_spacy(article_text)
        
        # Combine results
        analysis = {
            'title': title,
            'length': len(article_text),
            'regex_extraction': regex_results,
            'nltk_analysis': nltk_results,
            'spacy_analysis': spacy_results,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ Analysis complete!")
        return analysis

def main():
    """Demonstrate the news analyzer with sample articles"""
    
    # Sample news articles for demonstration
    articles = [
        {
            'title': "Tech Company Acquisition",
            'text': """
            Apple Inc. announced today that it will acquire the startup DataMind for $2.5 billion. 
            The deal, expected to close by December 15, 2024, will strengthen Apple's AI capabilities.
            
            DataMind, founded in 2019 by Dr. Sarah Chen and Mark Rodriguez in San Francisco, California,
            has developed revolutionary machine learning algorithms. The company's 150 employees will
            join Apple's AI division.
            
            "This acquisition represents a significant step forward in our AI strategy," said Apple CEO
            Tim Cook during a press conference. Investors reacted positively, with Apple's stock rising 3.2%.
            
            For more information, contact press@apple.com or visit https://apple.com/news.
            You can also call our investor relations at 408-996-1010.
            """
        },
        {
            'title': "Climate Change Report",
            'text': """
            The United Nations released its latest climate report on November 8, 2024, warning that
            global temperatures could rise by 2.5°C by 2050 if current trends continue.
            
            The report, compiled by 300 scientists from 50 countries, highlights the urgent need
            for action. "We are running out of time," said Dr. Maria Santos, lead author from
            the University of Cambridge.
            
            Key findings include:
            - Arctic ice loss accelerating at 15% per decade
            - Sea levels rising 3.2mm annually
            - Extreme weather events increasing by 45%
            
            The report recommends reducing carbon emissions by 50% within the next decade.
            Environmental groups praised the report, while some industry leaders expressed concerns
            about economic impacts.
            """
        }
    ]
    
    # Initialize analyzer
    analyzer = NewsAnalyzer()
    
    # Analyze each article
    results = []
    for article in articles:
        result = analyzer.analyze_article(article['text'], article['title'])
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total articles analyzed: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"\nArticle {i}: {result['title']}")
        print(f"  • Length: {result['length']} characters")
        print(f"  • Sentences: {result['nltk_analysis']['sentences']}")
        print(f"  • Sentiment: {max(result['nltk_analysis']['sentiment'], key=result['nltk_analysis']['sentiment'].get)}")
        
        if result['spacy_analysis']:
            entities = len(result['spacy_analysis']['entities'])
            print(f"  • Named entities: {entities}")
        
        if result['regex_extraction']:
            print(f"  • Structured data found: {list(result['regex_extraction'].keys())}")

if __name__ == "__main__":
    main() 