
import gradio as gr

from src.sampling_methods import sample_tokens
from src.sample_shake_speare_model import my_gpt, tokenizer


def complete_text(text, max_tokens_generated=100, temperature=1.0,freq_penalty=2):
    text_output = sample_tokens(my_gpt,
                                tokenizer,
                                text,
                                max_tokens_generated=max_tokens_generated,
                                temperature=temperature,
                                top_p=1,
                                freq_penalty=freq_penalty)
    return text_output

demo = gr.Interface(
    fn=complete_text,
    inputs=["text", 
        gr.Slider(10, 300, value = 100, step=1),
        gr.Slider(0, 3, value=1, step=0.1),
        gr.Slider(0, 10, value=0, step=0.1)],
    outputs=["text"],
)
demo.launch(server_name="0.0.0.0", server_port=5000)

