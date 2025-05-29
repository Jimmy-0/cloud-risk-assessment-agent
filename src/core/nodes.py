import json
from typing import Literal

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.types import Command

from src.core.llm import model, final_model
from src.core.state import AgentState
from src.db.db_query import generate_query, is_valid_query, query_summary
from src.db.db_setup import setup_database_connections
from src.utils.utils import (
    token_count,
    read_prompt,
    read_file_prompt,
    messages_token_count,
    get_latest_human_message,
    reasoning_prompt,
    trim_messages_to_max_tokens,
)

# Set up database connection for graph nodes
app_context = setup_database_connections()

SYSTEM_PROMPT = read_file_prompt("./src/prompts/report_system_prompt.txt")
VALID_REPORT_CATEGORIES = {"code", "container", "aws", "kubernetes", "all"}


def parse_report_command(input_string: str) -> str:
    """Extract the report category from a `/report` command."""
    command_prefix = "/report "

    if not input_string.startswith(command_prefix):
        raise ValueError("Input does not start with '/report'.")

    argument = input_string[len(command_prefix) :].strip()
    if not argument:
        raise ValueError("No argument provided after '/report'.")
    if argument not in VALID_REPORT_CATEGORIES:
        raise ValueError(
            f"Invalid argument '{argument}'. Allowed arguments are "
            f"{', '.join(VALID_REPORT_CATEGORIES)}."
        )
    return argument


async def classify_user_intent(state: AgentState):
    """Classify the user's query as either a report request or a regular question."""
    messages = state["messages"]
    query = get_latest_human_message(messages)
    print(f"\n\nUSER QUERY: {query} \n")

    try:
        category = parse_report_command(query)
        return Command(update={"category": category}, goto="summary")
    except ValueError:
        content = reasoning_prompt(
            "./src/prompts/intent_classification_prompt.txt", question=query
        )
        intent_response = await model.ainvoke([HumanMessage(content=content)])

        try:
            res = json.loads(intent_response.content)
            score = res.get("Score", 0)
            if score > 30:
                return Command(update={"intention": res, "user_query": query}, goto="querydb")
            else:
                return Command(update={"intention": res, "user_query": None}, goto="reason")
        except json.JSONDecodeError:
            print("Failed to parse intent classification response")
            return Command(update={"user_query": query}, goto="reason")


async def invoke_llm(state: AgentState):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def generate_summary_report(state: AgentState):
    print("--------------do_summary---------------")
    category = state["category"]

    summary_df, details_df = await query_summary(app_context.get_connection(), category)
    result = details_df.to_string(index=False)
    top5_result = details_df.to_string()
    summary = summary_df.to_string(index=False)

    template = read_prompt("summary")
    prompt = PromptTemplate(
        template=template, input_variables=["category", "summary", "result"]
    )
    formatted_prompt = prompt.format(category=category, summary=summary, result=result)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=formatted_prompt)]

    tokens = token_count(formatted_prompt)
    print(f"Token used: {tokens}\n")
    response = await final_model.ainvoke(messages)

    df_str = details_df.to_csv(index=False)
    return {
        "dataframe": df_str,
        "result_text": result,
        "top5": top5_result,
        "messages": [response],
    }


async def generate_insights(state: AgentState):
    print("--------------do_insight---------------")
    result = state["top5"]

    template = read_prompt("insight")
    prompt = PromptTemplate(template=template, input_variables=["result"])
    formatted_prompt = prompt.format(result=result)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=formatted_prompt)]

    response = await final_model.ainvoke(messages)
    return {"messages": [response]}


async def finalize_conclusion(state: AgentState):
    print("--------------do_conclude---------------")
    messages = state["messages"]
    result = state["result_text"]

    template = read_prompt("conclude")
    messages.append(HumanMessage(content=template))

    total_tokens = messages_token_count(messages)
    print(f"total message tokens: {total_tokens}")
    response = await final_model.ainvoke(messages)

    return {"messages": [HumanMessage(content=result), response]}


async def execute_db_query(state: AgentState) -> Command[Literal["reason"]]:
    messages = state["messages"]
    user_query = state["user_query"]
    category = state.get("category", "ALL").upper() if state.get("category") else "ALL"
    try:
        generated_query = await generate_query(user_query, category, model)
        if not is_valid_query(generated_query, app_context.get_engine()):
            print("Generated query is invalid or potentially unsafe.\n\n")
            return Command(update={"user_query": user_query}, goto="reason")
        print("Executing query...\n\n")
        cursor = app_context.get_connection().cursor()
        cursor.execute(generated_query)
        records = cursor.fetchall()
        if records:
            columns = [desc[0] for desc in cursor.description]
            results_str = "\n".join(str(dict(zip(columns, row))) for row in records)
        else:
            results_str = "No results returned."
        print("Query results prepared.\n\n")
        return Command(
            update={
                "user_query": user_query,
                "sql_query": generated_query,
                "query_results": results_str,
                "messages": messages + [SystemMessage(content="Query executed successfully.")],
            },
            goto="reason",
        )
    except Exception as e:
        print(f"Error during query execution: {e}\n\n")
        return Command(update={"user_query": user_query}, goto="reason")


async def provide_explanation(state: AgentState):
    try:
        user_query = state.get("user_query", "")
        sql_query = state.get("sql_query", "")
        query_results = state.get("query_results", "")
        messages = state.get("messages", [])

        if not user_query:
            user_query = get_latest_human_message(state["messages"])

        template = read_prompt("explanation")
        prompt = PromptTemplate(
            template=template, input_variables=["question", "sql_query", "scan_results"]
        )
        formatted_prompt = prompt.format(
            question=user_query, sql_query=sql_query, scan_results=query_results
        )

        if len(formatted_prompt) > 80000:
            formatted_prompt = formatted_prompt[:80000]

        messages.append(HumanMessage(content=formatted_prompt))
        messages = trim_messages_to_max_tokens(messages)
        explanation_response = await model.ainvoke(messages)

        return Command(
            update={
                "user_query": None,
                "sql_query": None,
                "query_results": None,
                "messages": state["messages"] + [explanation_response],
            }
        )
    except Exception as e:
        print(f"Error during explanation generation: {e}")
        return Command(
            update={
                "user_query": None,
                "sql_query": None,
                "query_results": None,
                "messages": state["messages"]
                + [SystemMessage(content="An error occurred while generating the explanation. Please try again.")],
            }
        )
