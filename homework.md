# Homework: Social Media Post Analyzer

## Objective
Build a social media post analyzer that extracts hashtags, mentions, and analyzes engagement patterns using RegEx, NLTK, and SpaCy.

## Task Description
Create a Python script that analyzes social media posts to:
1. Extract hashtags, mentions, and URLs using RegEx
2. Perform sentiment analysis and identify key topics with NLTK
3. Recognize named entities and analyze writing style with SpaCy

## Requirements

### Part 1: RegEx Extraction (30 points)
Create patterns to extract:
- **Hashtags**: #python, #AI, #MachineLearning
- **Mentions**: @username, @company
- **URLs**: http/https links
- **Emojis**: Unicode emoji patterns
- **Numbers**: Follower counts, likes, retweets

### Part 2: NLTK Analysis (40 points)
Implement:
- **Tokenization**: Split posts into words and sentences
- **Sentiment scoring**: Classify as positive, negative, or neutral
- **Keyword extraction**: Find most common meaningful words
- **Text statistics**: Calculate readability metrics

### Part 3: SpaCy Processing (30 points)
Add:
- **Named entity recognition**: Identify people, organizations, locations
- **Language detection**: Determine post language
- **Dependency analysis**: Find subject-verb-object relationships

## Sample Data
Use these sample social media posts for testing:

```python
sample_posts = [
    "Just launched our new #AI product! 🚀 Thanks to @team_awesome for the hard work. Check it out: https://oursite.com/product #innovation #tech",
    
    "Beautiful sunset in San Francisco today 🌅 #photography #california #nature Love this city! @visit_sf",
    
    "Breaking: Apple announces new iPhone with 50% better battery life! Stock up 12% 📈 #Apple #iPhone #tech #stocks",
    
    "Feeling grateful for 10,000 followers! 🙏 You all are amazing. Special thanks to @mentor_john for the guidance #milestone #grateful"
]
```

## Deliverables
1. **Python script** (`social_media_analyzer.py`) with your implementation
2. **Analysis report** (`analysis_report.md`) showing results for sample posts
3. **Code comments** explaining your RegEx patterns and NLP choices

## Evaluation Criteria
- **Correctness**: Patterns work accurately on sample data
- **Code quality**: Clean, readable, well-commented code
- **Analysis depth**: Meaningful insights from the text processing
- **Documentation**: Clear explanation of approach and results

## Submission Format
Create a folder named `social_media_analyzer` containing:
```
social_media_analyzer/
├── social_media_analyzer.py
├── analysis_report.md
├── requirements.txt
└── README.md
```

## Bonus Challenge (Extra Credit)
Extend your analyzer to:
- Detect trending topics across multiple posts
- Identify potential spam or bot accounts
- Analyze posting patterns and engagement correlation
- Create visualizations of your findings

## Due Date
Submit your completed assignment within one week of receiving this homework.

## Tips for Success
1. Start with simple RegEx patterns and test incrementally
2. Use NLTK's built-in datasets for comparison
3. Leverage SpaCy's pre-trained models effectively
4. Document your pattern choices and analysis decisions
5. Test edge cases and handle errors gracefully 