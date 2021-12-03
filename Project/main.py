# Prepare libraries
from flask import Flask, render_template, request
import functions as func
import pickle
import warnings

# Stop not important warnings and define the main flask application
warnings.filterwarnings("ignore")
main_application = Flask(__name__)

# Application home page
@main_application.route("/")
def index():
    return render_template("index.html", page_title="Article Summarization and Categorization")

# Analyze text page
# First we get the text from the input textarea
# Then get classifier and the number of sentences
# Get the language for calling the right model
# Get text summary and category
@main_application.route("/analyze_text", methods=['GET', 'POST'])
def analyze_text():
    if request.method == 'POST':
        input_text = request.form['text_input_text']
        classifier_model_name = request.form['text_classifier']
        summarizer_model_name = request.form['text_summarizer']
        sentences_number = request.form['text_sentences_number']
        classifier_model = pickle.load(open('models/' + classifier_model_name + '.pkl', 'rb'))
        text_summary, text_category = func.summarize_category(input_text, int(sentences_number), classifier_model, summarizer_model_name)
    return render_template("index.html", page_title="Article Summarization and Categorization", input_text=input_text, text_summary=text_summary, text_category=text_category)

if __name__ == "__main__":
    main_application.run()