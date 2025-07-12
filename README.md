# Processing Text in Python: NLTK, SpaCy, and Regular Expressions

## What is Text Processing?
Text processing is like teaching a computer to read and understand human language. Just like you can read a sentence and understand who did what, where, and when - we want computers to do the same thing.

**Why do we need it?**
- Chatbots need to understand what you're asking
- Email filters need to detect spam
- Social media platforms analyze sentiment (happy/sad posts)
- Search engines need to understand what you're looking for

## The Three Tools We'll Learn

Think of these tools like different types of reading glasses:
- **RegEx**: Finds specific patterns (like finding all phone numbers in a text)
- **NLTK**: Understands basic grammar and meaning (like a grammar teacher)
- **SpaCy**: Advanced understanding (like a literature professor)

---

## 1. Regular Expressions (RegEx) - The Pattern Finder

RegEx is like a super-powered "Find" function. Instead of finding exact words, it finds patterns.

### Simple Example: Finding Phone Numbers
```python
import re

text = "Call me at 555-123-4567 or email john@example.com"

# Find phone numbers (3 digits, dash, 3 digits, dash, 4 digits)
phone_pattern = r'\d{3}-\d{3}-\d{4}'
phones = re.findall(phone_pattern, text)
print(phones)  # Output: ['555-123-4567']
```

### What's happening?
- `\d` means "any digit (0-9)"
- `{3}` means "exactly 3 of the previous thing"
- `-` means "literal dash character"
- So `\d{3}-\d{3}-\d{4}` means "3 digits, dash, 3 digits, dash, 4 digits"

### More Practical Examples
```python
import re

text = """
Contact us at support@company.com or sales@company.com
Visit our website: https://www.company.com
Our office number is 555-123-4567
Founded in March 15, 2020
"""

# Find all email addresses
emails = re.findall(r'\w+@\w+\.\w+', text)
print("Emails:", emails)
# Output: ['support@company.com', 'sales@company.com']

# Find all websites
websites = re.findall(r'https?://[^\s]+', text)
print("Websites:", websites)
# Output: ['https://www.company.com']

# Find all dates (Month Day, Year)
dates = re.findall(r'\w+ \d{1,2}, \d{4}', text)
print("Dates:", dates)
# Output: ['March 15, 2020']
```

### When to Use RegEx
- Finding phone numbers, emails, URLs
- Extracting dates, prices, or other formatted data
- Cleaning messy text data
- Validating user input (like checking if an email looks correct)

---

## 2. NLTK - The Grammar Teacher

NLTK breaks down sentences like an English teacher would - finding subjects, verbs, and understanding the structure.

### Installation and Setup
```python
import nltk
# Download the tools we need (only do this once)
nltk.download('punkt')      # For splitting sentences
nltk.download('stopwords')  # For common words like "the", "and"
nltk.download('averaged_perceptron_tagger')  # For grammar analysis
```

### Example 1: Breaking Text into Pieces
```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello world! How are you today? I am learning Python."

# Split into sentences
sentences = sent_tokenize(text)
print("Sentences:")
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")

# Output:
# 1. Hello world!
# 2. How are you today?
# 3. I am learning Python.

# Split into words
words = word_tokenize(text)
print("\nWords:", words)
# Output: ['Hello', 'world', '!', 'How', 'are', 'you', 'today', '?', 'I', 'am', 'learning', 'Python', '.']
```

### Example 2: Finding Important Words
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog"
words = word_tokenize(text.lower())

# Remove common words that don't add much meaning
stop_words = set(stopwords.words('english'))
important_words = [word for word in words if word not in stop_words and word.isalpha()]

print("Original words:", words)
print("Important words:", important_words)
# Output: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

### Example 3: Understanding Grammar
```python
import nltk
from nltk.tokenize import word_tokenize

text = "The cat sits on the mat"
words = word_tokenize(text)

# Tag each word with its part of speech
pos_tags = nltk.pos_tag(words)
print("Word → Grammar Role:")
for word, tag in pos_tags:
    print(f"{word} → {tag}")

# Output:
# The → DT (Determiner)
# cat → NN (Noun)
# sits → VBZ (Verb, 3rd person singular)
# on → IN (Preposition)
# the → DT (Determiner)
# mat → NN (Noun)
```

### Example 4: Sentiment Analysis (Happy or Sad?)
```python
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

texts = [
    "I love this movie! It's amazing!",
    "This is the worst day ever.",
    "The weather is okay today."
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Sentiment: {scores}")
    print(f"Overall: {'Positive' if scores['compound'] > 0.1 else 'Negative' if scores['compound'] < -0.1 else 'Neutral'}")
    print()
```

---

## 3. SpaCy - The Literature Professor

SpaCy is like having a smart assistant that understands context and relationships in text.

### Installation
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Example 1: Finding People, Places, and Organizations
```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California. Tim Cook is the current CEO."

# Process the text
doc = nlp(text)

# Find named entities (people, places, companies)
print("Found these important things:")
for entity in doc.ents:
    print(f"'{entity.text}' is a {entity.label_}")

# Output:
# 'Apple Inc.' is a ORG (Organization)
# 'Steve Jobs' is a PERSON
# 'Cupertino' is a GPE (Geopolitical entity - place)
# 'California' is a GPE
# 'Tim Cook' is a PERSON
```

### Example 2: Understanding Word Relationships
```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The dog chased the cat quickly"
doc = nlp(text)

print("Word relationships:")
for token in doc:
    print(f"'{token.text}' depends on '{token.head.text}' ({token.dep_})")

# Output shows how words relate to each other:
# 'dog' is the subject of 'chased'
# 'cat' is the object of 'chased'
# 'quickly' describes how the chasing happened
```

### Example 3: Better Word Roots (Lemmatization)
```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The children were running and playing happily"
doc = nlp(text)

print("Original → Root form:")
for token in doc:
    if token.is_alpha:  # Only show actual words
        print(f"{token.text} → {token.lemma_}")

# Output:
# children → child
# were → be
# running → run
# playing → play
# happily → happily
```

---

## Putting It All Together: A Complete Example

Let's analyze a restaurant review using all three tools:

```python
import re
import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# Sample restaurant review
review = """
I visited Mario's Italian Restaurant last night with my family. 
The food was absolutely delicious! We ordered pizza and pasta. 
The service was excellent and the staff was very friendly. 
You can call them at 555-PIZZA-1 or visit their website at www.marios.com
I would definitely recommend this place! 5 stars!
"""

print("RESTAURANT REVIEW ANALYSIS")
print("=" * 50)

# 1. REGEX: Find contact information
print("\n1. CONTACT INFO (using RegEx):")
phones = re.findall(r'\d{3}-[A-Z]+-\d', review)
websites = re.findall(r'www\.[^\s]+', review)
print(f"Phone: {phones}")
print(f"Website: {websites}")

# 2. NLTK: Analyze sentiment
print("\n2. SENTIMENT ANALYSIS (using NLTK):")
sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(review)
print(f"Positive: {sentiment['pos']:.2f}")
print(f"Negative: {sentiment['neg']:.2f}")
print(f"Overall: {'POSITIVE' if sentiment['compound'] > 0.1 else 'NEGATIVE' if sentiment['compound'] < -0.1 else 'NEUTRAL'}")

# 3. SPACY: Find important entities
print("\n3. IMPORTANT ENTITIES (using SpaCy):")
nlp = spacy.load("en_core_web_sm")
doc = nlp(review)
for ent in doc.ents:
    print(f"'{ent.text}' → {ent.label_}")
```

**Output:**
```
RESTAURANT REVIEW ANALYSIS
==================================================

1. CONTACT INFO (using RegEx):
Phone: ['555-PIZZA-1']
Website: ['www.marios.com']

2. SENTIMENT ANALYSIS (using NLTK):
Positive: 0.45
Negative: 0.00
Overall: POSITIVE

3. IMPORTANT ENTITIES (using SpaCy):
'Mario's Italian Restaurant' → ORG
'last night' → TIME
'5' → CARDINAL
```

---

## Quick Reference: When to Use What

| Task | Tool | Example |
|------|------|---------|
| Find phone numbers | RegEx | `r'\d{3}-\d{3}-\d{4}'` |
| Find emails | RegEx | `r'\w+@\w+\.\w+'` |
| Check if text is happy/sad | NLTK | `SentimentIntensityAnalyzer()` |
| Break text into sentences | NLTK | `sent_tokenize()` |
| Find people's names | SpaCy | `doc.ents` where `label_ == 'PERSON'` |
| Find companies | SpaCy | `doc.ents` where `label_ == 'ORG'` |
| Get root form of words | SpaCy | `token.lemma_` |

## Practice Exercise

Try this with your own text:
1. Find all numbers in a text using RegEx
2. Count how many sentences are in the text using NLTK
3. Find all person names using SpaCy

```python
your_text = "Put any text here and experiment!"
# Your code here...
```

Remember: Start simple, then add complexity as you learn more! 