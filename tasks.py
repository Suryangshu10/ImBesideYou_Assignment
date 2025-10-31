# tasks.py
from crewai import Crew, Task
from agents import planner, retriever, summarizer, evaluator

# Define workflow for question/summary request
planner_task = Task(
    agent=planner,
    description="Plan the steps to summarize uploaded notes and generate revision cards."
)

retriever_task = Task(
    agent=retriever,
    description="Retrieve all document chunks relevant to the user's subject/topic."
)

summarizer_task = Task(
    agent=summarizer,
    description="Summarize the retrieved text into concise, user-friendly revision cards."
)

evaluator_task = Task(
    agent=evaluator,
    description="Evaluate the generated summary for relevance, coverage, clarity."
)

crew = Crew(
    agents=[planner, retriever, summarizer, evaluator],
    tasks=[planner_task, retriever_task, summarizer_task, evaluator_task],
)

def kickoff_pipeline(input_query, ground_truth):
    # Input can be user's question/input, full notes
    return crew.kickoff(inputs={"query": input_query, "ground_truth": ground_truth})
