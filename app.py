import torch
import gradio as gr
from unsloth import FastModel
from transformers import AutoTokenizer

# Load fine-tuned MedSaathi model
model, _ = FastModel.from_pretrained(
    model_name="FounderShubham1729/medsaathi-gemma4-e4b",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastModel.for_inference(model)

raw_tokenizer = AutoTokenizer.from_pretrained(
    "FounderShubham1729/medsaathi-gemma4-e4b"
)

SYSTEM = (
    "You are MedSaathi, a safe medicine assistant for Indian patients. "
    "Explain in Hinglish. Never give specific dose numbers. "
    "Always recommend confirming with a doctor or pharmacist. "
    "Warn about banned drugs FIRST, before anything else."
)

def ask(question, history):
    if not question.strip():
        return ""
    prompt = (
        f"<start_of_turn>user\n[System: {SYSTEM}]\n\n"
        f"{question}\n<end_of_turn>\n<start_of_turn>model\n"
    )
    inputs = raw_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1
        )
    return raw_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

with gr.Blocks(title="MedSaathi — Aapka Medicine Companion") as demo:
    gr.Markdown("""
    # 🏥 MedSaathi — Aapka Medicine Companion
    **Medicine explain karo | Jan Aushadhi savings | Banned drug alerts | Emergency first aid**
    *Hindi · English · Hinglish — jo comfortable ho*
    """)

    chatbot = gr.Chatbot(height=450, type="messages")
    msg = gr.Textbox(
        placeholder="Medicine ka naam ya sawaal likhein... (e.g. 'Dolo 650 kya hai?')",
        show_label=False
    )

    with gr.Row():
        submit = gr.Button("Bhejo →", variant="primary")
        clear  = gr.Button("Clear")

    gr.Examples(
        examples=[
            "Dolo 650 kya hai?",
            "Augmentin 625 ka sasta option?",
            "Corex cough syrup safe hai?",
            "Saanp ne kaata kya karein?",
            "Metformin ke baare mein batao",
            "Dog ne kaata, rabies ka darr hai",
            "Kitni tablet leni chahiye?",
        ],
        inputs=msg
    )

    def user(message, history):
        return "", history + [{"role": "user", "content": message}]

    def bot(history):
        response = ask(history[-1]["content"], history[:-1])
        history.append({"role": "assistant", "content": response})
        return history

    submit.click(user, [msg, chatbot], [msg, chatbot]).then(
        bot, chatbot, chatbot
    )
    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: [], None, chatbot)

    gr.Markdown("""
    ---
    ⚕️ *MedSaathi provides information only. Always confirm with your doctor or pharmacist.*
    🔬 *Fine-tuned Gemma 4 E4B | Unsloth QLoRA | 638 India-specific examples | Loss: 0.6003*
    """)

if __name__ == "__main__":
    demo.launch(share=True)
