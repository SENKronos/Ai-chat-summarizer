import argparse
import string
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data (run once)
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

def preprocess_sentence(sentence):
    """Preprocess a single sentence for TF-IDF."""
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence.strip()

def summarize_text(text, num_keypoints=5):
    """Extractive summarization using TF-IDF (exactly as per mission steps)."""
    if not text.strip():
        return ["No text provided to summarize."]

    # Split into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_keypoints:
        return [s.strip() for s in sentences]

    # Preprocess sentences for TF-IDF
    clean_sentences = [preprocess_sentence(s) for s in sentences]

    # Use TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)
        # Score each sentence by sum of TF-IDF scores
        sentence_scores = tfidf_matrix.sum(axis=1).A1
    except:
        # Fallback for very short text
        return [s.strip() for s in sentences[:num_keypoints]]

    # Get top scoring sentence indices (preserve original order)
    ranked_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_keypoints]
    ranked_indices.sort()

    summary = [sentences[i].strip() for i in ranked_indices]
    return summary

def main():
    parser = argparse.ArgumentParser(description="AI Chat Summarizer (Python + NLP)")
    parser.add_argument('files', nargs='+', help='One or more input text files to summarize')
    parser.add_argument('--num', '-n', type=int, default=5, help='Number of key points in summary (default: 5)')
    parser.add_argument('--output', '-o', help='Optional output file to save summary')
    
    args = parser.parse_args()

    # Read all input files and concatenate (supports single + multiple files)
    full_text = ""
    for file_path in args.files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text += f.read() + "\n\n"
            print(f"✅ Loaded: {file_path}")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

    if not full_text.strip():
        print("❌ No text to summarize.")
        return

    print("\n🤖 Generating summary using TF-IDF...")
    summary_points = summarize_text(full_text, args.num)

    # Output key points in bullet form
    print("\n=== 📋 SUMMARY (Key Points) ===")
    for point in summary_points:
        print(f"• {point}")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("=== AI Chat Summarizer Output ===\n\n")
            for point in summary_points:
                f.write(f"• {point}\n")
        print(f"\n💾 Summary saved to: {args.output}")

if __name__ == "__main__":
    main()