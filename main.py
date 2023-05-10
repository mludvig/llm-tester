#!/usr/bin/env python3

import os
import time
import argparse

from backends import backends

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Query LLMs")
    parser.add_argument(
        "-m",
        "--models",
        metavar="MODEL",
        dest="models",
        help="Model(s) to query, comma-separated. Prefix with - to exclude.",
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=0.1, help="Temperature"
    )
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens")
    args, prompts = parser.parse_known_args()

    # Validate and fix models
    available_models = [backend["name"] for backend in backends]
    if args.models is None:
        args.models = available_models
    else:
        model_args = args.models.split(",")
        selected_models = []
        deselected_models = []
        for model_arg in model_args:
            if model_arg.startswith("-"):
                deselected_models.append(model_arg[1:])
            else:
                selected_models.append(model_arg)
        if len(selected_models) > 0:
            args.models = selected_models
        else:
            args.models = available_models
        for deselected_model in deselected_models:
            if deselected_model in args.models:
                args.models.remove(deselected_model)
    for model in args.models:
        if model not in available_models:
            parser.error(f"Invalid model: {model}")
    print(f"Models: {args.models}")

    # Validate and fix prompts
    if prompts is None:
        parser.error("Prompt or prompt file(s) required")
    if os.path.isfile(prompts[0]):
        for prompt_file in prompts:
            if not os.path.isfile(prompt_file):
                parser.error(f"Invalid prompt file: {prompt_file}")
        args.prompt_files = prompts
    else:
        args.prompt = " ".join(prompts)

    return args


def query(models, prompt, temperature=0.1, max_tokens=200):
    print(f"===> Prompt:\n{prompt}")
    for model in models:
        backend = list(filter(lambda b: b["name"] == model, backends))[0]
        print(f"=> Model: {backend['name']}")
        ts = time.time()
        result, streaming = backend["query"](
            prompt, temperature, max_tokens, streaming=True
        )
        if not streaming:
            print(result)
        print(f"\n=> Done ({time.time()-ts:.2f}s):")
        print()
    print()


if __name__ == "__main__":
    args = parse_args()
    if args.prompt_files:
        for prompt_file in args.prompt_files:
            with open(prompt_file, "r") as f:
                prompt = f.read()
            query(args.models, prompt, args.temperature, args.max_tokens)
    else:
        query(args.models, args.prompt, args.temperature, args.max_tokens)
