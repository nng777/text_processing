# News Article Analyzer Project

## Overview
This project demonstrates practical text processing using RegEx, NLTK, and SpaCy by building a news article analyzer that extracts key information, performs sentiment analysis, and identifies named entities.

## What You'll Learn
- **Regular Expressions**: Pattern matching for structured data extraction
- **NLTK**: Basic NLP tasks like tokenization, stemming, and sentiment analysis
- **SpaCy**: Advanced NLP with named entity recognition and dependency parsing
- **Integration**: How to combine multiple text processing tools effectively

## Project Structure
```
news_analyzer.py     # Main analyzer class and demonstration
requirements.txt     # Python dependencies
project_guide.md     # This guide
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download SpaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 3. Run the Analyzer
```bash
python news_analyzer.py
```

## How It Works

### Phase 1: RegEx Extraction
The analyzer uses regular expressions to find structured information:
- **Emails**: `press@apple.com`
- **Phone numbers**: `408-996-1010`
- **URLs**: `https://apple.com/news`
- **Dates**: `December 15, 2024`
- **Money**: `$2.5 billion`
- **Percentages**: `3.2%`

**Key Concept**: RegEx excels at finding patterns in text. Each pattern is a template that matches specific formats.

### Phase 2: NLTK Analysis
NLTK performs fundamental NLP tasks:
- **Tokenization**: Splits text into sentences and words
- **Stop word removal**: Filters out common words like "the", "and"
- **Stemming**: Reduces words to root forms (running → run)
- **POS tagging**: Identifies parts of speech (noun, verb, adjective)
- **Sentiment analysis**: Determines emotional tone (positive/negative/neutral)

**Key Concept**: NLTK provides the building blocks for text analysis. You combine these tools to understand text structure and meaning.

### Phase 3: SpaCy Processing
SpaCy adds advanced NLP capabilities:
- **Named Entity Recognition**: Identifies people, organizations, locations
- **Lemmatization**: More accurate word root extraction than stemming
- **Dependency parsing**: Shows grammatical relationships between words
- **Pre-trained models**: Leverages machine learning for better accuracy

**Key Concept**: SpaCy uses machine learning models trained on large datasets to understand language context and meaning.

## Sample Output Analysis

When analyzing the tech acquisition article, you'll see:

### RegEx Finds:
- Money: `['$2.5 billion']`
- Email: `['press@apple.com']`
- Phone: `['408-996-1010']`
- URL: `['https://apple.com/news']`

### NLTK Discovers:
- Sentiment: POSITIVE (0.85)
- Common words: apple(3), compani(2), acquisit(2)
- POS tags: [('Apple', 'NNP'), ('Inc.', 'NNP'), ('announced', 'VBD')]

### SpaCy Identifies:
- Entities: [('Apple Inc.', 'ORG'), ('DataMind', 'ORG'), ('Dr. Sarah Chen', 'PERSON')]
- Dependencies: Apple --nsubj--> announced, DataMind --dobj--> acquire

## Key Learning Points

### 1. Tool Selection
- **RegEx**: Perfect for structured patterns (emails, dates, phone numbers)
- **NLTK**: Great for learning NLP concepts and research
- **SpaCy**: Best for production applications requiring speed and accuracy

### 2. Processing Pipeline
1. **Clean**: Remove noise and normalize text
2. **Extract**: Use RegEx for structured data
3. **Analyze**: Apply NLTK for basic NLP
4. **Understand**: Use SpaCy for advanced insights

### 3. Real-World Applications
- **Content moderation**: Detect inappropriate content
- **Information extraction**: Pull key facts from documents
- **Sentiment monitoring**: Track brand perception
- **Document classification**: Automatically categorize text

## Experiment Ideas
1. **Add new patterns**: Create RegEx for social media handles, stock symbols
2. **Custom sentiment**: Train your own sentiment classifier
3. **Entity linking**: Connect named entities to knowledge bases
4. **Text summarization**: Extract key sentences from articles
5. **Language detection**: Identify text language before processing

## Common Pitfalls
- **RegEx complexity**: Start simple, build complexity gradually
- **NLTK downloads**: Ensure all required data is downloaded
- **SpaCy models**: Different models for different languages/domains
- **Performance**: SpaCy is faster but uses more memory than NLTK

## Next Steps
- Explore transformer models (BERT, GPT) for advanced understanding
- Build a web interface for the analyzer
- Add support for multiple languages
- Create custom NLP models for specific domains 