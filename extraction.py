"""Step 1: RegEx Extraction (30 points)
1.Create patterns to extract:
1.1.Hashtags: #python, #AI, #MachineLearning
1.2.Mentions: @username, @company
1.3.URLs: http/https links
1.4.Emojis: Unicode emoji patterns
1.5.Numbers: Follower counts, likes, retweets

Step 2: NLTK Analysis (40 points)
2.Implement:
2.1.Tokenization: Split posts into words and sentences
2.2.Sentiment scoring: Classify as positive, negative, or neutral
2.3.Keyword extraction: Find most common meaningful words
2.4.Text statistics: Calculate readability metrics

Step 3: SpaCy Processing (30 points)
3.Add:
3.1.Named entity recognition: Identify people, organizations, locations
3.2.Language detection: Determine post language
3.3.Dependency analysis: Find subject-verb-object relationships

Sample Data
Use this sample social media posts for testing:

sample_posts = [
    "Just launched our new #AI product! 🚀 Thanks to @team_awesome for the hard work. Check it out: https://oursite.com/product #innovation #tech",
    "Beautiful sunset in San Francisco today 🌅 #photography #california #nature Love this city! @visit_sf",
    "Breaking: Apple announces new iPhone with 50% better battery life! Stock up 12% 📈 #Apple #iPhone #tech #stocks",
    "Feeling grateful for 10,000 followers! 🙏 You all are amazing. Special thanks to @mentor_john for the guidance #milestone #grateful"
]

Deliverables
1.Python script (social_media_analyzer.py) with your implementation
2.Analysis report (analysis_report.md) showing results for sample posts
3.Code comments explaining your RegEx patterns and NLP choices"""