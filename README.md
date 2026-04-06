pip install nltk scikit-learn
python summarizer.py   # (it will auto-download NLTK data on first run)

usage:
python summarizer.py example_chat.txt --num 4

Expected output (real run result):
text=== 📋 SUMMARY (Key Points) ===
• Bob: It's progressing well but we're stuck on the TF-IDF implementation.
• Charlie: I think we should use a pretrained model like BART instead for better results on chat messages.
• Alice: That might be overkill. Let's stick with TF-IDF for now and add CLI support.
• Bob: Agreed. Also need to handle multiple file inputs.