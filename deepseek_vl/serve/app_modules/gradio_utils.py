from functools import wraps

import gradio as gr


def wrap_gen_fn(gen_fn):
    @wraps(gen_fn)
    def wrapped_gen_fn(prompt, *args, **kwargs):
        try:
            yield from gen_fn(prompt, *args, **kwargs)
        except gr.Error as g_err:
            raise g_err
        except Exception as e:
            raise gr.Error(f'Failed to generate text: {e}') from e

    return wrapped_gen_fn


def delete_last_conversation(chatbot, history):
    if len(history) % 2 != 0:
        gr.Error("history length is not even")
        return (
            chatbot,
            history,
            "Delete Done",
        )

    if len(chatbot) > 0:
        chatbot.pop()

    if len(history) > 0 and len(history) % 2 == 0:
        history.pop()
        history.pop()

    return (
        chatbot,
        history,
        "Delete Done",
    )


def reset_state():
    return [], [], None, "Reset Done"


def reset_textbox():
    return gr.update(value=""), ""


def cancel_outputing():
    return "Stop Done"


def transfer_input(input_text, input_image):
    print("transferring input text and input image")
    return (
        input_text,
        input_image,
        gr.update(value=""),
        gr.update(value=None),
        gr.Button(visible=True),
    )


class State:
    interrupted = False

    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False


shared_state = State()
