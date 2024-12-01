import os
import traceback
import pandas as pd
import gradio as gr
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    verbose=True,
)

def query_dataframe(file, question, history):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file.name)
    except pd.errors.EmptyDataError:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "The uploaded file is empty."})
        return history
    except pd.errors.ParserError:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "The uploaded file is not a valid CSV."})
        return history
    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"Error loading file: {e}"})
        return history

    # Create the pandas agent
    try:
        p_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            max_iterations=10,
            max_execution_time=120,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
            agent_executor_kwargs={"handle_parsing_errors": True},
            allow_dangerous_code=True
        )
    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"Error creating agent: {e}"})
        return history

    # Run the agent with the input question
    try:
        # Using invoke instead of run
        response = p_agent.invoke(
            {"input": question},
            config={"handle_parsing_errors": True}
        )["output"]
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
    except ValueError as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"An error occurred: {e}"})
    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"Unexpected error: {e}\n{traceback.format_exc()}"})

    return history

# Define the Gradio UI with a chat history
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            <style>
            .gradio-container {
                background-color: #fffbe7;
                font-family: Arial, sans-serif;
            }
            .header {
                font-size: 24px;
                font-weight: bold;
                color: #333;
                text-align: center;
                padding: 10px;
            }
            </style>
            """
        )

        gr.Markdown("<div class='header'>CSV Query Interface</div>")

        # CSV file uploader
        csv_file = gr.File(label="Upload CSV file", file_types=[".csv"])

        # Text box for entering the question
        question = gr.Textbox(
            label="Enter your question about the CSV data",
            placeholder="Type your question here..."
        )

        # Button to submit the question
        query_button = gr.Button("Submit", variant="primary")

        # Chatbot for displaying responses in a chat format
        response_display = gr.Chatbot(label="Chat History", type="messages")

        # Initialize an empty history variable
        history = gr.State([])

        # Bind the update_response function to the Submit button
        query_button.click(
            query_dataframe,
            inputs=[csv_file, question, history],
            outputs=[response_display]
        )

    return app

if __name__ == "__main__":
    app = create_ui()
    try:
        app.launch(share=True)
    except Exception as e:
        print(f"Could not create share link: {e}")
        print("Launching local-only version...")
        app.launch(share=False)
