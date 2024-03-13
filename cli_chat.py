# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# -*- coding: utf-8 -*-

import argparse
import os
import sys
from threading import Thread

import torch
from PIL import Image
from transformers import TextIteratorStreamer

from deepseek_vl.utils.io import load_pretrained_model


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def get_help_message(image_token):
    help_msg = (
        f"\t\t DeepSeek-VL-Chat is a chatbot that can answer questions based on the given image. Enjoy it! \n"
        f"Usage: \n"
        f"    1. type `exit` to quit. \n"
        f"    2. type `{image_token}` to indicate there is an image. You can enter multiple images, "
        f"e.g '{image_token} is a dot, {image_token} is a cat, and what is it in {image_token}?'. "
        f"When you type `{image_token}`, the chatbot will ask you to input image file path. \n"
        f"    4. type `help` to get the help messages. \n"
        f"    5. type `new` to start a new conversation. \n"
        f"    Here is an example, you can type: '<image_placeholder>Describe the image.'\n"
    )

    return help_msg


@torch.inference_mode()
def response(
    args, conv, pil_images, tokenizer, vl_chat_processor, vl_gpt, generation_config
):
    prompt = conv.get_prompt()
    prepare_inputs = vl_chat_processor.__call__(
        prompt=prompt, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_config["inputs_embeds"] = inputs_embeds
    generation_config["attention_mask"] = prepare_inputs.attention_mask
    generation_config["streamer"] = streamer

    thread = Thread(target=vl_gpt.language_model.generate, kwargs=generation_config)
    thread.start()

    yield from streamer


def get_user_input(hint: str):
    user_input = ""
    while user_input == "":
        try:
            user_input = input(f"{hint}")
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            user_input = "exit"

    return user_input


def chat(args, tokenizer, vl_chat_processor, vl_gpt, generation_config):
    image_token = vl_chat_processor.image_token
    help_msg = get_help_message(image_token)

    while True:
        print(help_msg)

        pil_images = []
        conv = vl_chat_processor.new_chat_template()
        roles = conv.roles

        while True:
            # get user input
            user_input = get_user_input(
                f"{roles[0]} [{image_token} indicates an image]: "
            )

            if user_input == "exit":
                print("Chat program exited.")
                sys.exit(0)

            elif user_input == "help":
                print(help_msg)

            elif user_input == "new":
                os.system("clear")
                pil_images = []
                conv = vl_chat_processor.new_chat_template()
                torch.cuda.empty_cache()
                print("New conversation started.")

            else:
                conv.append_message(conv.roles[0], user_input)
                conv.append_message(conv.roles[1], None)

                # check if the user input is an image token
                num_images = user_input.count(image_token)
                cur_img_idx = 0

                while cur_img_idx < num_images:
                    try:
                        image_file = input(
                            f"({cur_img_idx + 1}/{num_images}) Input the image file path: "
                        )
                        image_file = (
                            image_file.strip()
                        )  # trim whitespaces around path, enables drop-in from for example Dolphin

                    except KeyboardInterrupt:
                        print()
                        continue

                    except EOFError:
                        image_file = None

                    if image_file and os.path.exists(image_file):
                        pil_image = load_image(image_file)
                        pil_images.append(pil_image)
                        cur_img_idx += 1

                    elif image_file == "exit":
                        print("Chat program exited.")
                        sys.exit(0)

                    else:
                        print(
                            f"File error, `{image_file}` does not exist. Please input the correct file path."
                        )

                # get the answer by the model's prediction
                answer = ""
                answer_iter = response(
                    args,
                    conv,
                    pil_images,
                    tokenizer,
                    vl_chat_processor,
                    vl_gpt,
                    generation_config,
                )
                sys.stdout.write(f"{conv.roles[1]}: ")
                for char in answer_iter:
                    answer += char
                    sys.stdout.write(char)
                    sys.stdout.flush()

                sys.stdout.write("\n")
                sys.stdout.flush()
                conv.update_last_message(answer)
                # conv.messages[-1][-1] = answer


def main(args):
    # setup
    tokenizer, vl_chat_processor, vl_gpt = load_pretrained_model(args.model_path)
    generation_config = dict(
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=args.max_gen_len,
        use_cache=True,
    )
    if args.temperature > 0:
        generation_config.update(
            {
                "do_sample": True,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty,
            }
        )
    else:
        generation_config.update({"do_sample": False})

    chat(args, tokenizer, vl_chat_processor, vl_gpt, generation_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/deepseek-vl-7b-chat",
        help="the huggingface model name or the local path of the downloaded huggingface model.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_gen_len", type=int, default=512)
    args = parser.parse_args()
    main(args)
