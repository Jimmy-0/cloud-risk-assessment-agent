from io import StringIO
import json
from typing import Dict, Optional

import pandas as pd
import chainlit as cl
from chainlit.server import app
from fastapi import APIRouter, HTTPException, Response
from starlette.routing import BaseRoute, Route
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig

from src.core.workflow import graph, REASONING_NODE
from src.core import nodes


@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
    """Authenticate users via header information."""
    return cl.User(identifier="admin", metadata={"role": "admin", "provider": "header"})


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(msg: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    chat_history.append({"role": "user", "content": msg.content})
    config = {"configurable": {"thread_id": msg.thread_id}}

    final_answer = cl.Message(content="")
    async for m, metadata in graph.astream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[], **config),
    ):
        if (
            m.content
            and not isinstance(m, HumanMessage)
            and not isinstance(m, SystemMessage)
            and metadata["langgraph_node"] in REASONING_NODE
        ):
            await final_answer.stream_token(m.content)

        if (
            "finish_reason" in m.response_metadata
            and m.response_metadata["finish_reason"] == "stop"
        ):
            await final_answer.stream_token("\n\n")

        if (
            "finish_reason" in m.response_metadata
            and m.response_metadata["finish_reason"] == "stop"
            and metadata["langgraph_node"] in ["insight"]
        ):
            state = graph.get_state(config=config)
            df_str = state.values["dataframe"]
            df = pd.read_csv(StringIO(df_str))
            elements = [cl.Dataframe(data=df, display="inline", name="Dataframe")]
            await cl.Message(content="Report Table:", elements=elements).send()

    await final_answer.send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="All scan executive summary report", message="/report all", icon="/public/ranking.png"),
        cl.Starter(label="Kubernetes executive summary report", message="/report kubernetes", icon="/public/k8s.png"),
        cl.Starter(label="AWS executive summary report", message="/report aws", icon="/public/aws.png"),
        cl.Starter(label="Code executive summary report", message="/report code", icon="/public/code.png"),
        cl.Starter(label="Container executive summary report", message="/report container", icon="/public/container.png"),
    ]


@cl.on_chat_resume
async def on_chat_resume(thread):
    cl.user_session.set("chat_history", [])

    if thread.get("metadata") is not None:
        metadata = thread["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        if metadata.get("chat_history") is not None:
            state_messages = []
            chat_history = metadata["chat_history"]
            for message in chat_history:
                cl.user_session.get("chat_history").append(message)
                if message["role"] == "user":
                    state_messages.append(HumanMessage(content=message["content"]))
                else:
                    state_messages.append(AIMessage(content=message["content"]))

            thread_id = thread["id"]
            config = {"configurable": {"thread_id": thread_id}}
            state = graph.get_state(config).values
            if "messages" not in state:
                state["messages"] = state_messages
                graph.update_state(config, state)


cust_router = APIRouter()


@cust_router.get("/blob/{object_key}")
async def serve_blob_file(object_key: str):
    if nodes.app_context.storage_client is None:
        raise HTTPException(status_code=500, detail="Storage client not initialized")
    file_data = await nodes.app_context.storage_client.download_file(object_key)
    return Response(content=file_data, media_type="application/octet-stream")


serve_route: list[BaseRoute] = [r for r in app.router.routes if isinstance(r, Route) and r.name == "serve"]
for route in serve_route:
    app.router.routes.remove(route)

app.include_router(cust_router)
app.router.routes.extend(serve_route)
