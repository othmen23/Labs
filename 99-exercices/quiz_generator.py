import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question text")
    options: list[str] = Field(description="Four possible answers")
    correct_answer: str = Field(description="The correct option, must match one of 'options'")
    difficulty: str = Field(description="One of: easy, medium, hard")


structured_llm = llm.with_structured_output(QuizQuestion)

quiz_template = ChatPromptTemplate.from_template(
    """Generate one multiple-choice quiz question about: {topic}.

Don't repeat the previously generated questions:
{history}

Generate exactly four possible answers.
The correct_answer must exactly match one of the options.
"""
)

quiz_gen = quiz_template | structured_llm


def print_quiz(q: QuizQuestion):
    print(f"\nQ: {q.question}")

    for i, opt in enumerate(q.options, start=1):
        print(f"   {i}. {opt}")

    print(f"Answer: {q.correct_answer}")
    print(f"Difficulty: {q.difficulty}")


@tool
def save_quiz_markdown(filename: str, topic: str, q: QuizQuestion) -> str:
    """Save one quiz question to a Markdown file."""

    if not filename.endswith(".md"):
        filename += ".md"

    file_exists = os.path.exists(filename)

    with open(filename, "a", encoding="utf-8") as file:

        if not file_exists:
            file.write(f"# Quiz: {topic}\n\n")

        file.write(f"## Question\n\n")
        file.write(f"**{q.question}**\n\n")

        for i, option in enumerate(q.options, start=1):
            file.write(f"{i}. {option}\n")

        file.write(f"\n**Correct answer:** {q.correct_answer}\n")
        file.write(f"**Difficulty:** {q.difficulty}\n\n")
        file.write("---\n\n")

    return f"Question saved in {filename}"


while True:

    topic = input("\nInsert a topic: ")

    number_questions = int(input("How many questions do you want? "))

    gen_questions = []

    for i in range(number_questions):

        print(f"\nGenerating question {i + 1}/{number_questions}...")

        quiz = quiz_gen.invoke(
            {
                "topic": topic,
                "history": gen_questions
            }
        )

        print_quiz(quiz)

        gen_questions.append(quiz)

        save = input("\nDo you want to save this question? (y/n): ").lower()

        if save == "y":

            filename = input("Enter the Markdown filename: ")

            result = save_quiz_markdown.invoke(
                {
                    "filename": filename,
                    "topic": topic,
                    "q": quiz
                }
            )

            print(result)

    print(f"\nFinished generating {number_questions} questions about {topic}.")

    new_topic = input("\nDo you want to choose another topic? (y/n): ").lower()

    if new_topic != "y":
        print("\nQuiz generator finished.")
        break