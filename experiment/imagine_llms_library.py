import time
from io import BytesIO
from openai import OpenAI
from google.genai import types
import pandas as pd
from google import genai
import anthropic
import argparse
from PIL import Image
import base64

TEXT_FAIL = "Text generation failed! Retrying..."
IMAGE_FAIL = "Image generation failed! Retrying..."
RESPONSE_TEXT = (
    "What object best represents what you imagine? Respond in at most two words."
)
RESPONSE_IMAGE_TEXT = (
    "What object best represents the image you generated? Respond in at most two words."
)
MODEL_CONS = "Model:"
INS1 = "Ins 1"
INS2 = "Ins 2"
INS3 = "Ins 3"
INS4 = "Ins 4"
INS_LIST = [INS1, INS2, INS3, INS4]


def determine_family_from_model(model: str) -> str:
    """
    Determine the family of the model based on its name.

    :param model: str: The name of the model.
    :type model: str
    :return: str: The family of the model, either "gemini", "openai", or "claude".
    :rtype: str
    """
    match model:
        case "gemini-2.5-pro-preview-05-06":
            return "gemini"
        case "gemini-2.0-flash-preview-image-generation":
            return "gemini"
        case "gemini-2.0-flash-exp-image-generation":  # old
            return "gemini"
        case "gemini-2.5-flash-preview-native-audio-dialog":
            return "gemini"
        case "gemini-2.0-flash":
            return "gemini"
        case "claude-opus-4-1-20250805":
            return "claude"
        case "claude-sonnet-4-20250514":
            return "claude"
        case "o3-2025-04-16":
            return "openai"
        case "o4-mini-2025-04-16":
            return "openai"
        case "gpt-4.1-2025-04-14":
            return "openai"
        case "o3-pro-2025-06-10":
            return "openai"
        case "chatgpt-4o-latest":
            return "openai"
        case "gpt-5-2025-08-07":
            return "openai"
    return ""


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    Use for instructions.

    :param data_path: str: Path to the CSV file containing the data.
    :type data_path: str
    :return: pd.DataFrame: DataFrame containing the loaded data.
    :rtype: pd.DataFrame
    """
    return pd.read_csv(data_path)


def build_gemini_client(api_path: str) -> genai.Client:
    """
    Build a Gemini client using the API key from the specified path.

    :param api_path: str: Path to the file containing the Gemini API key.
    :type api_path: str
    :return: genai.Client: A Gemini client instance.
    :rtype: genai.Client
    """
    with open(api_path, "r") as f:
        api_key = (
            f.read().strip()
        )  # Ensure the API key is stripped of whitespace and newlines
    return genai.Client(api_key=api_key)


def build_gemini_chat(model: str, client: genai.Client) -> list:
    """
    Build a Gemini chat session with the specified model and client.

    This is a fairly arbitrary struct list that contains 1: the new chat, 2: the string of the model, 3:
    the client, 4: a None value to be replaced by previous message information.

    :param model: str: The model to use for the chat session.
    :type model: str
    :param client: genai.Client: The Gemini client instance.
    :type client: genai.Client
    :return: list: A list containing the chat session, model name, client, and None (the last message ID).
    :rtype: list
    """
    return [client.chats.create(model=model), model, client, None]
    # 'chat' is a list where [gemini chat obj, model name str, gemini client obj, last message id str]


def build_openai_chat(model: str, client: OpenAI) -> list:
    """
    Build an OpenAI chat session with the specified model and client.

    :param model: str: The model to use for the chat session.
    :type model: str
    :param client: OpenAI: The OpenAI client instance.
    :type client: OpenAI
    :return: list: A list containing the chat session, model name, client, and None (the last response ID).
    :rtype: list
    """
    return [[], model, client, None]
    # 'chat' is a list where [list of previous messages, model name str, openai client obj, last message id str]


def build_claude_chat(model: str, client: anthropic.Anthropic) -> list:
    """
    Build a Claude chat session with the specified model and client.

    :param model: str: The model to use for the chat session.
    :type model: str
    :param client: anthropic.Anthropic: The Claude client instance.
    :type client: anthropic.Anthropic
    :return: list: A list containing the chat session, model name, client.
    :rtype: list
    """
    return [[], model, client]
    # 'chat' is a list [list of previous messages, model name str, anthropic client obj]


def build_openai_client(api_path: str, org: str, project: str) -> OpenAI:
    """
    Build an OpenAI client using the API key from the specified path.

    :param api_path: str: Path to the file containing the OpenAI API key. This should be a secret.
    :type api_path: str
    :param org: str: The OpenAI organization ID.
    :type org: str
    :param project: str: The OpenAI project ID.
    :type project: str
    :return: OpenAI: An OpenAI client instance.
    :rtype: OpenAI
    """
    with open(api_path, "r") as f:
        api_key = (
            f.read().strip()
        )  # Ensure the API key is stripped of whitespace and newlines
    return OpenAI(api_key=api_key, organization=org, project=project)


def build_claude_client(api_path: str) -> anthropic.Anthropic:
    """
    Build a Claude client using the API key from the specified path.

    :param api_path: str: Path to the file containing the Claude API key. This should be a secret.
    :type api_path: str
    :return: anthropic.Anthropic: A Claude client instance.
    :rtype: anthropic.Anthropic
    """
    with open(api_path, "r") as f:
        api_key = (
            f.read().strip()
        )  # Ensure the API key is stripped of whitespace and newlines
    return anthropic.Anthropic(api_key=api_key)


def hang_and_wait() -> str:
    """
    Hang and wait for user confirmation to proceed after an exception.
    :return: str: User input to confirm whether to continue or quit.
    :rtype: str
    """
    print("Waiting for user to view exception...")
    return input("Confirm exception...?")


def confirm_exception() -> str:
    """
    Confirm with the user whether to continue or quit after an exception.

    :return: str: User input to confirm whether to continue or quit.
    :rtype: str
    """
    confirmation = hang_and_wait()
    if "quit" in confirmation.lower():
        print("Exiting due to user request.")
        return "quit"
    else:
        return confirmation.lower()


def get_thinking_token_count_claude(model: str) -> int:
    """
    Get the maximum token count for a given Claude model.

    :param model: str: The name of the Claude model.
    :type model: str
    :return: int: The maximum token count for the model.
    :rtype: int
    """
    match model:
        case "claude-opus-4-1-20250805":
            return 9000
        case "claude-sonnet-4-20250514":
            return 4000
    return 0  # default max token count


def get_max_token_count_claude(model: str) -> int:
    """
    Get the maximum token count for a given Claude model.

    :param model: str: The name of the Claude model.
    :type model: str
    :return: int: The maximum token count for the model.
    :rtype: int
    """
    match model:
        case "claude-opus-4-1-20250805":
            return 10000
        case "claude-sonnet-4-20250514":
            return 5000
    return 0  # default max token count


def generate_claude_text(prompt, chat, n_exceptions=0, hang=0):
    """
    Generate a Claude text output from a given prompt through a given chat.

    :param prompt: str: the prompt to send to the model
    :type prompt: str
    :param chat: list: the chat session containing the Claude client and model.
    :type chat: list
    :param n_exceptions: int: the number of exceptions encountered so far. Default is 0.
    :param hang: time to wait before retrying in case of an exception. Default is 0.
    :return: the output text
    """
    if n_exceptions > 5: # only allow 5 exceptions before requiring user input
        if "quit" in confirm_exception():
            return None
    else:
        time.sleep(hang)
    try:
        chat[0] += [{"role": "user", "content": prompt}]
        # chat 0: prompt history
        response = chat[2].messages.create(
            model=chat[1],
            messages=chat[0],
            max_tokens=get_max_token_count_claude(chat[1]),
            thinking={
                "type": "enabled",
                "budget_tokens": get_thinking_token_count_claude(chat[1]),
            },
        )
    except Exception as e: # message fails to generate or similar
        print(e)
        time.sleep(60)
        return generate_claude_text(prompt, chat, n_exceptions + 1, hang) # recursive retry
    try:
        text = ""
        for block in response.content:
            if block.type == "thinking":
                print("Thinking:", block.thinking) # thinking summary not actual tokens
                print()
            if block.type == "text":
                text = block.text
    except Exception as e: # response generation failed, retry recursively
        print(e)
        print(TEXT_FAIL)
        time.sleep(60)
        return generate_claude_text(prompt, chat, n_exceptions + 1, hang)
    if text == "":  # check to make sure that the text is not empty
        print(TEXT_FAIL)
        time.sleep(60)
        return generate_claude_text(prompt, chat, n_exceptions + 1, hang)
    chat[0] += [{"role": "assistant", "content": response.content}]
    return text


def generate_gemini_image(
    prompt: str, chat: list, n_exceptions: int = 0, hang: int = 0
) -> tuple:
    """
    Generate an image using the Gemini model.

    :param prompt: str: The prompt for image generation.
    :type prompt: str
    :param chat: list: The chat session containing the Gemini client and model.
    :type chat: list
    :param n_exceptions: int: The number of exceptions encountered so far. Default is 0.
    :type n_exceptions: int
    :param hang: int: The time to wait before retrying in case of an exception. Default is 0.
    :type hang: int
    :return: tuple: A tuple containing the generated text and image data.
    :rtype: tuple
    """
    if n_exceptions > 5:  # faily arbitrary number that can be changed as desired.
        # useful to prevent excessive failed requests which can still cost money
        if "quit" in confirm_exception():
            return None, None  # bad output tuple
    else:
        time.sleep(
            hang
        )  # if the model crashed, we want to wait for a small amount of time
        # to prevent repeated crashes
    try:
        response = chat[0].send_message(
            prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
    except Exception as e:  # Catching all exceptions to handle retries
        # This should probably be modified to specific exception types
        print(e)
        time.sleep(60)
        return generate_gemini_image(prompt, chat, n_exceptions + 1, hang)
    text = ""
    image = None
    try:  # if this try fails, then the model failed to generate proper output
        for part in response.candidates[0].content.parts:
            if part.text is not None:  # language output
                text = part.text
            elif part.inline_data is not None:  # image output
                image = BytesIO(part.inline_data.data)
            # one part cannot be both, one output can contain both
    except Exception as e:
        print(e)
        time.sleep(60)
        return generate_gemini_image(prompt, chat, n_exceptions + 1, hang)
    if text == "" and image is None:
        print("Text/image generation failed! Retrying...")
        time.sleep(60)
        return generate_gemini_image(prompt, chat, n_exceptions + 1, hang)
    return text, image


def generate_gemini_text(prompt, chat, n_exceptions=0, hang=0):
    """
    Generate a Gemini text output from a given prompt through a given chat.

    :param prompt: str: the prompt to send to the model
    :type prompt: str
    :param chat: list: the chat session containing the Gemini client and model.
    :type chat: list
    :param n_exceptions: int: the number of exceptions encountered so far. Default is 0.
    :param hang: time to wait before retrying in case of an exception. Default is 0.
    :return: the output text
    """
    if n_exceptions > 5:
        if "quit" in confirm_exception():
            return None
    else:
        time.sleep(hang)
    try:
        response = chat[0].send_message(
            prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_modalities=(
                    ["TEXT"]
                    if chat[1] != "gemini-2.0-flash-preview-image-generation"
                    else ["TEXT", "IMAGE"]
                    # the new gemini flash preview model crashes when only given the text modality
                    # the old version does not, this fix isn't perfect, but it works
                ),
            ),
        )
    except Exception as e:
        print(e)
        time.sleep(60)
        return generate_gemini_text(prompt, chat, n_exceptions + 1, hang)
    try:
        text = response.text  # if response.text dne this excepts
    except Exception as e:
        print(e)
        print(TEXT_FAIL)
        time.sleep(60)
        return generate_gemini_text(prompt, chat, n_exceptions + 1, hang)
    if text == "":  # check to make sure that the text is not empty
        print(TEXT_FAIL)
        time.sleep(60)
        return generate_gemini_text(prompt, chat, n_exceptions + 1, hang)
    return text


def generate_openai_text(
    prompt, chat, n_exceptions=0, hang=0, reasoning=True, reasoning_level="high"
):
    """
    Generate an OpenAI text output from a given prompt through a given chat.

    :param prompt: str: the prompt to send to the model
    :type prompt: str
    :param chat: list: the chat session containing the Gemini client and model.
    :type chat: list
    :param n_exceptions: int: the number of exceptions encountered so far. Default is 0.
    :type n_exceptions: int
    :param hang: time to wait before retrying in case of an exception. Default is 0.
    :type hang: int
    :param reasoning: bool: whether to use reasoning in the model. Default is True.
    :type reasoning: bool
    :param reasoning_level: str: the level of reasoning. Default is "high".
    :return: the output text
    """
    if n_exceptions > 5:
        if "quit" in confirm_exception():
            return None, None  # bad output
    else:
        time.sleep(hang)
    prompt_struct = {  # imput prompt struct to the responses API
        "role": "user",
        "content": prompt,
    }
    chat[0] = [prompt_struct]
    try:
        if not reasoning:
            response = chat[2].responses.create(  # chat 2: chat
                model=chat[1],  # chat 1: string model name
                input=chat[0],  # chat 0: prompt
                previous_response_id=chat[
                    3
                ],  # chat 3: previous message ID, can be None
            )
        else:
            response = chat[2].responses.create(
                model=chat[1],
                input=chat[0],
                previous_response_id=chat[3],
                reasoning={"effort": reasoning_level},
            )
    except Exception as e:
        print(e)
        time.sleep(60)
        return generate_openai_text(
            prompt, chat, n_exceptions + 1, hang, reasoning, reasoning_level
        )
    try:
        text = response.output_text
        if reasoning:
            if response.reasoning.summary: # if there is a reasoning summary print it out
                # none of the models used create these properly
                print("Reasoning Summary:", response.reasoning.summary)
                print()
        r_tokens = response.usage.output_tokens_details.reasoning_tokens # get the reasoning token count for usage
    except Exception as e: # generation failed
        print(e)
        print(TEXT_FAIL)
        time.sleep(60)
        return generate_openai_text(
            prompt, chat, n_exceptions + 1, hang, reasoning, reasoning_level
        )
    if text == "":
        print(TEXT_FAIL) # no text generated
        time.sleep(60)
        return generate_openai_text(
            prompt, chat, n_exceptions + 1, hang, reasoning, reasoning_level
        )
    chat[3] = response.id  # set previous message ID to current message
    return text, r_tokens


def generate_openai_image(
    prompt, chat, n_exceptions=0, hang=0, reasoning=True, reasoning_level="high"
):
    """
    Generate an OpenAI image/text output from a given prompt through a given chat.

    :param prompt: str: the prompt to send to the model
    :param chat: list: the chat session containing the Gemini client and model.
    :param n_exceptions: int: the number of exceptions encountered so far. Default is 0.
    :param hang: int: the time to wait before retrying in case of an exception. Default is 0.
    :param reasoning: bool: whether to use reasoning in the model. Default is True.
    :param reasoning_level: str: the level of reasoning. Default is "high".
    :return: the output text and image as a tuple
    """
    if n_exceptions > 5:
        if "quit" in confirm_exception():
            return None, None
    else:
        time.sleep(hang)
    prompt_struct = {
        "role": "user",
        "content": prompt,
    }
    chat[0] = [prompt_struct]  # .append(prompt_struct)
    try:
        if not reasoning: # reasoning with images, vs without
            response = chat[2].responses.create(
                model=chat[1],
                input=chat[0],
                tools=[
                    {"type": "image_generation"}
                ],  # enable the image_generation tool
                previous_response_id=chat[3],
            )
            image_data = [  # get the image data
                output.result
                for output in response.output
                if output.type == "image_generation_call"
            ]
        else:
            response = chat[2].responses.create(
                model=chat[1],
                input=chat[0],
                tools=[{"type": "image_generation"}],
                previous_response_id=chat[3],
                reasoning={
                    "effort": reasoning_level,
                },
            )
            image_data = [
                output.result
                for output in response.output
                if output.type == "image_generation_call"
            ]
    except Exception as e: # image or text failed here, can happen if the image is explicit
        print(e)
        print(IMAGE_FAIL)
        time.sleep(60)
        return generate_openai_image(prompt, chat, n_exceptions + 1, hang)
    try:
        text = response.output_text  # get the output text, this can be ""
    except Exception as e:
        print(e)
        print(IMAGE_FAIL)
        time.sleep(60)
        return generate_openai_image(prompt, chat, n_exceptions + 1, hang)
    if (
        text == "" and not image_data
    ):  # make sure at least one of an image or output text exists
        print(TEXT_FAIL)
        time.sleep(60)
        return generate_openai_image(prompt, chat, n_exceptions + 1, hang)
    chat[3] = response.id
    return text, image_data  # return the text and the image data


def save_data(response_dict: dict, out_path: str):
    """
    Save the response data to a given path.

    :param response_dict: dict: the response data
    :param out_path: str: the output path
    :return: the new dataframe
    """
    response_df = pd.DataFrame.from_dict(response_dict, orient="index")
    response_df.to_csv(out_path)
    return response_df


def build_models(
    models, api_path_openai=None, api_path_gemini=None, api_path_claude=None
):
    """
    Build a list of OpenAI and Gemini chats

    :param models: list strings of OpenAI and Gemini model names
    :param api_path_openai: OpenAI API key path
    :param api_path_gemini: Gemini API key path
    :param api_path_claude: Claude API key path
    :return: the lists of OpenAI and Gemini chats
    """
    if api_path_openai is None and api_path_gemini is None and api_path_claude is None:
        raise ValueError("At least one API path must be provided.")
    openai_chats = []
    gemini_chats = []
    claude_chats = []
    for model in models: # go through each model name
        family = determine_family_from_model(model) # figure out the family
        if family == "gemini":
            client = build_gemini_client(api_path_gemini) # build the corresponding client
            chat = build_gemini_chat(model, client) # build the corresponding chat
            gemini_chats.append(chat) # add it to the list
        elif family == "openai":
            client = build_openai_client(
                api_path_openai,
                org="org",
                project="proj",
            )
            chat = build_openai_chat(model, client)
            openai_chats.append(chat)
        elif family == "claude":
            client = build_claude_client(api_path_claude)
            chat = build_claude_chat(model, client)
            claude_chats.append(chat)
    return openai_chats, gemini_chats, claude_chats


def run_instructions_images(
    row, openai_chats, gemini_chats, num, reasoning=False, reasoning_level="high"
):
    """
    Iterate over the instruction set of a given row of instructions
    (i.e. a dict) using the given chats for image generation.

    :param row: the row of instructions
    :type row: dict
    :param openai_chats: list of OpenAI chats
    :type openai_chats: list
    :param gemini_chats: list of Gemini chats
    :type gemini_chats: list
    :param num: identifer for naming images
    :type num: str
    :param reasoning: bool: whether to use reasoning in the model. Default is True.
    :type reasoning: bool
    :param reasoning_level: str: the level of reasoning. Default is "high".
    :type reasoning_level: str
    :return: output tuple of response lists for each model
    """
    openai_responses = []
    gemini_responses = []

    for x in INS_LIST:  # iterate over Ins 1, Ins 2, Ins 3, Ins 4
        if pd.isna(row[1][x]):  # stop when instructions stop
            break
        prompt = str(row[1][x]).replace(
            "<X>",
            (
                "Generate an image with"
                if x == INS1  # primary or consecutive instructions have different forms
                else "From there, modify the image with"
            ),
        )
        print("Prompt:", prompt)
        for chat in openai_chats:
            text, image = generate_openai_image(
                prompt, chat, reasoning=reasoning, reasoning_level=reasoning_level
            )
            if image:
                image_base64 = image[0]
                with open(
                    f"output_images/{chat[1]}_{num}_{x.replace(" ", "")}.png", "wb"
                ) as f:  # save the image file give the model type and the identifier
                    f.write(base64.b64decode(image_base64)) # openai output is b64
            else:
                print(IMAGE_FAIL) # no image
            print(MODEL_CONS, text)
        for chat in gemini_chats:
            text, image = generate_gemini_image(prompt, chat)
            print(MODEL_CONS, text)
            if image:
                image = Image.open(image) # gemini output is PIL compatible
                image.save(f"output_images/{chat[1]}_{num}_{x.replace(" ", "")}.png")
            else:
                print(IMAGE_FAIL) # no image
    print("Final Prompt:", RESPONSE_IMAGE_TEXT)
    # Final prompt is only language, no images
    for chat in openai_chats:
        response = generate_openai_text(  # OpenAI
            RESPONSE_IMAGE_TEXT,
            chat,
            reasoning=reasoning,
            reasoning_level=reasoning_level,
        )
        print(MODEL_CONS, response)
        openai_responses.append(response)
    for chat in gemini_chats:
        response = generate_gemini_text(RESPONSE_IMAGE_TEXT, chat)  # Gemini
        print(MODEL_CONS, response)
        gemini_responses.append(response)

    return openai_responses, gemini_responses # no claude images


def run_instructions(
    row,
    openai_chats,
    gemini_chats,
    claude_chats,
    reasoning=False,
    reasoning_level="high",
):
    """
    Iterate over the instruction set of a given row of instructions
    (i.e. a dict) using the given chats for text generation.

    :param row:
    :param openai_chats:
    :param gemini_chats:
    :param reasoning:
    :param reasoning_level:
    :return:
    """
    openai_responses = []
    gemini_responses = []
    claude_responses = []

    openai_usage = []

    for x in INS_LIST:
        if pd.isna(row[1][x]):
            continue
        prompt = str(row[1][x]).replace(
            "<X>",
            "Imagine" if x == INS1 else "From there, imagine",
            # Variation on imagine, first vs. consecutive
        )
        print("Prompt:", prompt)
        for chat in openai_chats:
            output, r_tokens = generate_openai_text(
                prompt, chat, reasoning=reasoning, reasoning_level=reasoning_level
            )
            print(MODEL_CONS, output)
            openai_usage.append(r_tokens)
        for chat in gemini_chats:
            print(MODEL_CONS, generate_gemini_text(prompt, chat))
        for chat in claude_chats:
            print(MODEL_CONS, generate_claude_text(prompt, chat))
    print("Final Prompt:", RESPONSE_TEXT)
    for chat in openai_chats:
        response, r_tokens = generate_openai_text(
            RESPONSE_TEXT, chat, reasoning=reasoning, reasoning_level=reasoning_level
        )
        print(MODEL_CONS, response)
        openai_responses.append(response)
        openai_usage.append(r_tokens)
    for chat in gemini_chats:
        response = generate_gemini_text(RESPONSE_TEXT, chat)
        print(MODEL_CONS, response)
        gemini_responses.append(response)
    for chat in claude_chats:
        response = generate_claude_text(RESPONSE_TEXT, chat)
        print(MODEL_CONS, response)
        claude_responses.append(response)

    return (openai_responses, openai_usage), gemini_responses, claude_responses


def reset_chats(openai_chats=None, gemini_chats=None, claude_chats=None):
    """Reset contexts of the given chats"""
    print("Resetting chats...")
    if openai_chats is not None:
        for chat in openai_chats: # openai we reset the list and give no previous message id to reset context
            chat[0] = []
            chat[3] = None
    if gemini_chats is not None: # gemini we build a whole new chat to reset context
        for n, chat in enumerate(gemini_chats):
            gemini_chats[n] = build_gemini_chat(chat[1], chat[2])
    if claude_chats is not None:
        for n, chat in enumerate(claude_chats): # same for claude as gemini
            claude_chats[n] = build_claude_chat(chat[1], chat[2])


def iterate_instructions(
    prompt_df,
    openai_chats,
    gemini_chats,
    claude_chats,
    single_context=False,
    reasoning=False,
    reasoning_level="high",
):
    """Iterate over a given instruction list row-by-row with given settings."""
    print(reasoning_level)
    openai_responses, gemini_responses, claude_responses = [], [], []
    openai_usage = []
    for row in prompt_df.iterrows():
        if not single_context:  # i.e. if MULTIPLE context variant reset chats
            reset_chats(openai_chats, gemini_chats, claude_chats)
        new_openai_tuple, new_gemini_responses, new_claude_responses = run_instructions(
            row,
            openai_chats,
            gemini_chats,
            claude_chats,
            reasoning=reasoning,
            reasoning_level=reasoning_level, # the tuple contains output text as well as token usage data
        )

        new_openai_responses, new_openai_usage = new_openai_tuple # usage data is reasoning response token count

        openai_usage.append(new_openai_usage) # append a list instead of extending to separate blocks

        openai_responses.extend(new_openai_responses)
        gemini_responses.extend(new_gemini_responses)
        claude_responses.extend(new_claude_responses)
    return (openai_responses, openai_usage), gemini_responses, claude_responses


def iterate_instructions_images(
    prompt_df,
    openai_chats,
    gemini_chats,
    single_context=False,
    reasoning=False,
    reasoning_level="high",
):
    """Iterate over a given instruction list row-by-row in image generation format."""
    openai_responses, gemini_responses = [], [] # no claude images
    for n, row in enumerate(prompt_df.iterrows()):
        if not single_context:  # i.e. if MULTIPLE context variant reset chats
            reset_chats(openai_chats, gemini_chats, None)
        new_openai_responses, new_gemini_responses = run_instructions_images(
            row,
            openai_chats,
            gemini_chats,
            f"{n}_single" if single_context else f"{n}_multiple",
            reasoning=reasoning,
            reasoning_level=reasoning_level,
        )
        openai_responses.extend(new_openai_responses)
        gemini_responses.extend(new_gemini_responses)
    return openai_responses, gemini_responses


def main(
    models,
    data_path,
    out_path,
    api_path_openai,
    api_path_gemini,
    api_path_claude,
    context_variant=0,
    images=False,
    reasoning=False,
    reasoning_level="high",
):
    """
    Run and save data for any number of different LLMs (presuming they have proper cases added to the relevant locations

    Implements the main experiment of
    Artificial Phantasia: Evidence for Propositional Reasoning-Based Mental Imagery in Large Language Models
    """

    if context_variant not in [0, 1, 2]:
        raise ValueError(
            "Invalid context variant. Choose from 0 (SINGLE), 1 (MULTIPLE), or 2 (BOTH)."
        )
    print(
        "Context Variant:",
        (
            "SINGLE"
            if context_variant == 0
            else "MULTIPLE" if context_variant == 1 else "BOTH"
        ),
    )
    openai_chats, gemini_chats, claude_chats = build_models(
        models, api_path_openai, api_path_gemini, api_path_claude
        # we could have been given a whole bunch of models to run
    )
    prompt_df = pd.read_csv(data_path)
    reset_chats(openai_chats, gemini_chats, claude_chats) # prelimary reset to be super safe
    openai_responses_sc, gemini_responses_sc, claude_responses_sc = None, None, None # need to initialize for the if
    openai_responses_mc, gemini_responses_mc, claude_responses_mc = None, None, None # statement later
    openai_tuple_sc, openai_tuple_mc = None, None
    if not images: # no images, mc and sc
        if context_variant == 0 or context_variant == 2: # SINGLE
            openai_tuple_sc, gemini_responses_sc, claude_responses_sc = (
                iterate_instructions(
                    prompt_df,
                    openai_chats,
                    gemini_chats,
                    claude_chats,
                    True,
                    reasoning=reasoning,
                    reasoning_level=reasoning_level,
                )
            )
        reset_chats(openai_chats, gemini_chats, claude_chats) # reset chats inbetween running context variants
        if context_variant == 1 or context_variant == 2: # MULTIPLE
            openai_tuple_mc, gemini_responses_mc, claude_responses_mc = (
                iterate_instructions(
                    prompt_df,
                    openai_chats,
                    gemini_chats,
                    claude_chats,
                    False,
                    reasoning=reasoning,
                    reasoning_level=reasoning_level,
                )
            )
    else: # no images makes things easier
        if context_variant == 0 or context_variant == 2: # SINGLE
            openai_responses_sc, gemini_responses_sc = iterate_instructions_images(
                prompt_df, openai_chats, gemini_chats, True, reasoning, reasoning_level
            )
        reset_chats(openai_chats, gemini_chats, claude_chats)
        openai_responses_mc, gemini_responses_mc, claude_responses_mc = None, None, None
        if context_variant == 1 or context_variant == 2: # MULTIPLE
            openai_responses_mc, gemini_responses_mc = iterate_instructions_images(
                prompt_df, openai_chats, gemini_chats, False, reasoning, reasoning_level
            )

    openai_usage_sc, openai_usage_mc = None, None

    if openai_tuple_mc is not None:
        openai_responses_mc, openai_usage_mc = openai_tuple_mc
    if openai_tuple_sc is not None:
        openai_responses_sc, openai_usage_sc = openai_tuple_sc

    usage_dict = {} # construct the usage dict and then save it
    if openai_usage_sc:
        usage_dict["openai_usage_sc"] = openai_usage_sc
    if openai_usage_mc:
        usage_dict["openai_usage_mc"] = openai_usage_mc
    save_data(usage_dict, out_path.replace(".csv", "_usage.csv"))

    response_dict = {} # print and built a response dictionary with all the responses
    if openai_responses_sc:
        print("OpenAI Single Context Responses:", openai_responses_sc)
        response_dict["openai_sc"] = openai_responses_sc
    if gemini_responses_sc:
        print("Gemini Single Context Responses:", gemini_responses_sc)
        response_dict["gemini_sc"] = gemini_responses_sc
    if claude_responses_sc:
        print("Claude Single Context Responses:", claude_responses_sc)
        response_dict["claude_sc"] = claude_responses_sc
    if openai_responses_mc:
        print("OpenAI Multiple Context Responses:", openai_responses_mc)
        response_dict["openai_mc"] = openai_responses_mc
    if gemini_responses_mc:
        print("Gemini Multiple Context Responses:", gemini_responses_mc)
        response_dict["gemini_mc"] = gemini_responses_mc
    if claude_responses_mc:
        print("Claude Multiple Context Responses:", claude_responses_mc)
        response_dict["claude_mc"] = claude_responses_mc
    save_data(response_dict, out_path) # save data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Imagine LLMs Library")
    parser.add_argument(
        "--models", nargs="+", required=True, help="List of model names to use"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the input data CSV file"
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to save the output CSV file"
    )
    parser.add_argument(
        "--api_path_openai",
        type=str,
        required=False,
        help="Path to OpenAI API key file",
    )
    parser.add_argument(
        "--api_path_gemini",
        type=str,
        required=False,
        help="Path to Gemini API key file",
    )
    parser.add_argument(
        "--api_path_claude",
        type=str,
        required=False,
        help="Path to Claude API key file",
    )
    parser.add_argument(
        "--context_variant",
        type=int,
        default=2,
        help="Context variant for instruction processing",
    )
    parser.add_argument(
        "--images",
        required=False,
        type=bool,
        default=False,
        help="Whether to generate images instead of text",
    )
    parser.add_argument(
        "--reasoning",
        required=False,
        type=bool,
        default=False,
        help="Whether to use reasoning in OpenAI responses",
    )
    parser.add_argument(
        "--reasoning_level",
        required=False,
        type=str,
        default="high",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning level for OpenAI responses",
    )
    args = parser.parse_args()
    main(**vars(args))
