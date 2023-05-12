import os

# set up openai, cohere, ai21, etc
from langchain.llms import OpenAI, Cohere, AI21, HuggingFaceHub, GPT4All
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# as of now GPT4all doesn't work with langchain
from pygpt4all import GPT4All as pyGPT4All
from pygpt4all import GPT4All_J as pyGPT4All_J

callbacks = [StreamingStdOutCallbackHandler()]


def engine_openai(model, prompt, temperature, max_tokens, streaming):
    llm = OpenAI(
        model_name=model,
        max_tokens=max_tokens,
        temperature=temperature,
        callbacks=callbacks,
        streaming=streaming,
    )
    result = llm(prompt)
    return result, streaming


def engine_openai_chat(model, prompt, temperature, max_tokens, streaming):
    llm = ChatOpenAI(
        model_name=model,
        max_tokens=max_tokens,
        temperature=temperature,
        callbacks=callbacks,
        streaming=streaming,
    )
    result = llm([HumanMessage(content=prompt)])
    return result.content.strip(), streaming


def engine_cohere(model, prompt, temperature, max_tokens, streaming):
    llm = Cohere(temperature=temperature)
    llm.model = model
    llm.max_tokens = max_tokens
    result = llm(prompt)
    return result, False


def engine_ai21(model, prompt, temperature, max_tokens, streaming):
    llm = AI21(temperature=temperature)
    llm.model = model
    llm.maxTokens = max_tokens
    result = llm(prompt)
    return result, False


def engine_hf(model, prompt, temperature, max_tokens, streaming):
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={"temperature": temperature, "max_length": max_tokens},
    )
    result = llm(prompt)
    return result, False


## Langchain doesn't work with GPT4all right now
def engine_gpt4all_langchain(model, prompt, temperature, max_tokens, streaming):
    model_dir = os.getenv("GPT4ALL_MODEL_DIR")
    llm = GPT4All(
        # model=f"{model_dir}/ggml-gpt4all-j-v1.3-groovy.bin",
        # backend='gptj',
        model=f"{model_dir}/{model}",
        callbacks=callbacks,
        verbose=True,
        temp=temperature)
    result = llm(prompt)
    return result, False


def engine_gpt4all(model, prompt, temperature, max_tokens, streaming):
    model_dir = os.getenv("GPT4ALL_MODEL_DIR")
    model_name = f"{model_dir}/{model}"
    model = pyGPT4All(model_name)
    result = ""
    for token in model.generate(prompt, temp=temperature):
        result += token
        if streaming:
            print(token, end="", flush=True)
    if streaming:
        print()
    return result, streaming


def engine_gpt4all_j(model, prompt, temperature, max_tokens, streaming):
    model_dir = os.getenv("GPT4ALL_MODEL_DIR")
    model_name = f"{model_dir}/{model}"
    model = pyGPT4All_J(model_name)
    result = ""
    for token in model.generate([prompt], temp=temperature):
        result += token
        if streaming:
            print(token, end="", flush=True)
    if streaming:
        print()
    return result, streaming


backends = [
    {
        "name": "davinci",
        "engine": engine_openai,
        "model": "text-davinci-003",
    },
    {
        "name": "gpt-3.5",
        "engine": engine_openai_chat,
        "model": "gpt-3.5-turbo",
    },
    {
        "name": "gpt-4",
        "engine": engine_openai_chat,
        "model": "gpt-4",
    },
    {
        "name": "cohere",
        "engine": engine_cohere,
        "model": "command",
    },
    {
        "name": "ai21",
        "engine": engine_ai21,
        "model": "j2-jumbo-instruct",
    },
    # {"name": "hf", "query": query_hf},
    {
        "name": "vicuna",
        "engine": engine_gpt4all_langchain,
        "model": "ggml-vicuna-13b-1.1-q4_2.bin",
    },
    {
        "name": "stable-vicuna",
        "engine": engine_gpt4all_j,
        "model": "ggml-stable-vicuna-13B.q4_2.bin",
    },
    {
        "name": "snoozy",
        "engine": engine_gpt4all,
        "model": "ggml-gpt4all-l13b-snoozy.bin",
    },
]
