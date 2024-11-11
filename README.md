Text Summarizer Model
A deep learning-based text summarization model that generates concise summaries from longer pieces of text. This project uses a customized sequence-to-sequence language model and various sampling techniques to create high-quality summaries, scoring optimally on ROUGE metrics.

Table of Contents
Features
Demo
Installation
Usage
Results
Technical Details
Evaluation
Project Structure
Contributing
License
Features
Efficient Text Summarization: Uses transformer-based architecture for text summarization.
Sampling Techniques: Includes Top-K and Temperature Sampling for diverse output generation.
Flexible Customization: Adjustable parameters like summary length, temperature, and top-K values.
Scoring: Evaluates results with ROUGE metrics (ROUGE-1, ROUGE-2, and ROUGE-L) to assess summary quality.
Demo
Input:

text
Copy code
The quick brown fox jumps over the lazy dog. This is a simple sentence that demonstrates the use of all the letters in the English alphabet. It is often used in typing exercises and font displays to showcase the style and clarity of a font. Many other fonts are used around the world for different purposes...
Generated Summary:

text
Copy code
The quick brown fox is a sentence that demonstrates the use of all the letters in the alphabet, commonly used in typing exercises and font displays.
Installation
Prerequisites
Python 3.8+
PyTorch 1.8+
Hugging Face Transformers
Rouge-score
Installation Steps
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/text_summarizer.git
cd text_summarizer
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Alternatively, you can manually install the necessary packages:

bash
Copy code
pip install torch transformers rouge-score
Usage
Running Text Summarization
Summarize a Text Document:

python
Copy code
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Advik-7/text_summarizer")
model = AutoModelForSeq2SeqLM.from_pretrained("Advik-7/text_summarizer")

input_text = """
The quick brown fox jumps over the lazy dog. This is a simple sentence that demonstrates the use of all the letters in the English alphabet...
"""

# Tokenize and generate summary
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=50,
    min_length=20,
    top_k=50,
    temperature=1.5,
    do_sample=True
)

# Decode and print summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Summary:", summary)
Testing with ROUGE Score:

python
Copy code
from rouge_score import rouge_scorer

reference_summary = "A concise version of the text explaining the use of fonts."
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, summary)
print(scores)
Parameters
max_length: Maximum length of the generated summary.
min_length: Minimum length of the summary.
top_k: Controls diversity by limiting token selection to the top k options.
temperature: Adjusts randomness. Higher values (e.g., 1.5) produce more creative summaries.
Results
The model has been tested on various datasets and performs well on short-form texts. ROUGE scores (for example use cases) are as follows:

ROUGE-1: 0.56 Precision, 0.38 Recall, 0.45 F1
