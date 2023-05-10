import os

# set up openai, cohere, ai21, etc
from langchain.llms import OpenAI, Cohere, AI21, HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# as of now GPT4all doesn't work with langchain
from pygpt4all import GPT4All, GPT4All_J

callbacks = [StreamingStdOutCallbackHandler()]


def query_openai_davinci(prompt, temperature, max_tokens, streaming=False):
    llm = OpenAI(
        model_name="text-davinci-003",
        max_tokens=max_tokens,
        temperature=temperature,
        callbacks=callbacks,
        streaming=streaming,
        verbose=True,
    )
    result = llm(prompt)
    return result, streaming


def query_openai_turbo(prompt, temperature, max_tokens, streaming=False):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     max_tokens=max_tokens,
                     temperature=temperature,
                     callbacks=callbacks,
                     streaming=streaming,
                     verbose=True)
    result = llm([HumanMessage(content=prompt)])
    return result.content.strip(), streaming


def query_openai_gpt4(prompt, temperature, max_tokens, streaming=False):
    llm = ChatOpenAI(model_name="gpt-4",
                     max_tokens=max_tokens,
                     temperature=temperature,
                     callbacks=callbacks,
                     streaming=streaming,
                     verbose=True)
    result = llm([HumanMessage(content=prompt)])
    return result.content.strip(), streaming


def query_cohere(prompt, temperature, max_tokens, streaming=False):
    llm = Cohere(temperature=temperature)
    llm.model = "command"
    llm.max_tokens = max_tokens
    result = llm(prompt)
    return result, False


def query_ai21(prompt, temperature, max_tokens, streaming=False):
    llm = AI21(temperature=temperature,
            #    callbacks=callbacks,
            #    streaming=streaming,
               verbose=True)
    llm.model = "j2-jumbo-instruct"
    llm.maxTokens = max_tokens
    result = llm(prompt)
    return result, False


def query_hf(prompt, temperature, max_tokens, streaming=False):
    #repo_id = "EleutherAI/gpt-neo-2.7B"
    repo_id = "databricks/dolly-v2-12b"
    llm = HuggingFaceHub(repo_id=repo_id,
                         model_kwargs={"temperature":temperature,
                                       "max_length":max_tokens},
                         )
    result = llm(prompt)
    return result, False


# def query_gpt4all_langchain(prompt, temperature, max_tokens):
#     model_dir = os.getenv("GPT4ALL_MODEL_DIR")
#     llm = GPT4All(
#         # model=f"{model_dir}/ggml-gpt4all-j-v1.3-groovy.bin",
#         # backend='gptj',
#         model=f"{model_dir}/ggml-gpt4all-l13b-snoozy.bin",
#         callbacks=callbacks,
#         verbose=True,
#         temp=temperature)
#     result = llm(prompt)
#     return result


def query_gpt4all(prompt, temperature, max_tokens, streaming=False):
    model_dir = os.getenv("GPT4ALL_MODEL_DIR")
    model_name = f"{model_dir}/ggml-gpt4all-l13b-snoozy.bin"
    model = GPT4All(model_name)
    result = ""
    for token in model.generate(prompt, temp=temperature):
        result += token
        if streaming:
            print(token, end='', flush=True)
    if streaming:
        print()
    return result, streaming


def query_gpt4all_j(prompt, temperature, max_tokens, streaming=False):
    model_dir = os.getenv("GPT4ALL_MODEL_DIR")
    model_name = f"{model_dir}/ggml-gpt4all-l13b-snoozy.bin"
    # model_name = f"{model_dir}/ggml-gpt4all-j-v1.3-groovy.bin"
    model = GPT4All_J(model_name)
    result = ""
    for token in model.generate([prompt], temp=temperature):
        result += token
        if streaming:
            print(token, end='', flush=True)
    if streaming:
        print()
    return result, streaming


backends = [
    {
        "name": "davinci",
        "query": query_openai_davinci
    },
    {
        "name": "gpt-3.5-turbo",
        "query": query_openai_turbo
    },
    {
        "name": "gpt-4",
        "query": query_openai_gpt4
    },
    {
        "name": "cohere",
        "query": query_cohere
    },
    {
        "name": "ai21",
        "query": query_ai21
    },
    {
        "name": "hf",
        "query": query_hf
    },
    {
        "name": "gpt4all",
        "query": query_gpt4all
    },
    {
        "name": "gpt4all-j",
        "query": query_gpt4all_j
    },
]