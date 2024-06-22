from __future__ import annotations

from typing import Literal, Union
from collections.abc import Iterable
from pydantic import BaseModel
from instructor import Mode
import pytest
import instructor


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


def test_sync_parallel_tools__error(client):
    client = instructor.from_openai(client, mode=Mode.PARALLEL_TOOLS)

    with pytest.raises(TypeError):
        resp = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You must always use tools"},
                {
                    "role": "user",
                    "content": "What is the weather in toronto and dallas and who won the super bowl?",
                },
            ],
            response_model=Weather,
        )


def test_sync_no_tool_calls(client):
    client = instructor.from_openai(client, mode=Mode.PARALLEL_TOOLS)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You NEVER USE tools. Just give an answer"},
            {
                "role": "user",
                "content": "Who won the super bowl?",
            },
        ],
        response_model=Iterable[Weather],
    )
    response_list = list(resp)

    assert isinstance(response_list[0], str)
    assert len(response_list) == 1


@pytest.mark.asyncio
async def test_async_no_tool_calls(aclient):
    client = instructor.from_openai(aclient, mode=Mode.PARALLEL_TOOLS)
    resp = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You NEVER USE tools. Just give an answer"},
            {
                "role": "user",
                "content": "Who won the super bowl?",
            },
        ],
        response_model=Iterable[Weather],
    )
    response_list = list(resp)
    assert isinstance(response_list[0], str)
    assert len(response_list) == 1


def test_sync_parallel_tools_or(client):
    client = instructor.from_openai(client, mode=Mode.PARALLEL_TOOLS)
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    assert len(list(resp)) == 3


@pytest.mark.asyncio
async def test_async_parallel_tools_or(aclient):
    client = instructor.from_openai(aclient, mode=Mode.PARALLEL_TOOLS)
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl? Do 3 seperate calls",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    assert len(list(resp)) == 3


def test_sync_parallel_tools_one(client):
    client = instructor.from_openai(client, mode=Mode.PARALLEL_TOOLS)
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas?",
            },
        ],
        response_model=Iterable[Weather],
    )
    assert len(list(resp)) == 2
