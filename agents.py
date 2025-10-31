# agents.py
from crewai import Agent, Task
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Setup base LLM (replace MODEL_NAME, API_KEY)
llm = OpenAI(model="gpt-3.5-turbo", openai_api_key="YOUR_KEY")

# Planner Agent
planner = Agent(
    role="Planner",
    goal="Decompose user's note summarization request into clear steps.",
    backstory="Knows how to break down academic tasks and guide retrieval.",
    tools=[], # Only high-level guidance here
    llm=llm,
)

# Retriever Agent (RAG)
db = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings(openai_api_key="YOUR_KEY"))
def retrieve_docs(query):
    return db.similarity_search(query)

retriever = Agent(
    role="Retriever",
    goal="Fetch most relevant document chunks for the summarizer.",
    backstory="Has access to all user's notes indexed in the vector database.",
    tools=[retrieve_docs],
    llm=llm,
)

# Summarizer Agent (fine-tuned model via LoRA/PEFT)
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
lora_model = PeftModel.from_pretrained(base_model, "path/to/lora/adapter")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
summarizer_pipe = pipeline("summarization", model=lora_model, tokenizer=tokenizer)

def summarize_text(inputs):
    return summarizer_pipe(inputs)[0]['summary_text']

summarizer = Agent(
    role="Summarizer",
    goal="Generate readable summaries and flashcards from retrieved text.",
    backstory="Specialized in academic summarization, fine-tuned via LoRA.",
    tools=[summarize_text],
    llm=None,  # Directly calls the summarization pipeline
)

# Evaluation Agent
def evaluate_summary(summary, ground_truth):
    # Simple ROUGE metric, implement your own if needed
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary, ground_truth)
    return scores

evaluator = Agent(
    role="Evaluator",
    goal="Evaluate summary quality, coverage, and clarity.",
    backstory="Expert in academic output evaluation using established metrics.",
    tools=[evaluate_summary],
    llm=llm,
)
