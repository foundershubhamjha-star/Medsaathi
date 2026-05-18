# MedSaathi 🏥 — Aapka Medicine Companion

> *My mother takes 6 medicines. One day a pharmacist suggested a substitute 
> I didn't recognise. I used AI to verify it — same drug, half the price. 
> But I had to know how to ask. 1.4 billion people can't. That's MedSaathi.*

## What It Does
- 💊 Explains medicines in simple Hinglish
- 💰 Shows Jan Aushadhi alternatives (saves 70–90%)
- 🚨 CDSCO banned drug alerts (including April 2025 bans)
- 🔍 Pharmacist substitute verification
- 🏥 Emergency first aid — snakebite, rabies, burns, CPR, dog bite

## Live Demo
🔗 [Try MedSaathi](https://huggingface.co/spaces/FounderShubham1729/medsaathi)

## Model
🤗 [Fine-tuned Model](https://huggingface.co/FounderShubham1729/medsaathi-gemma4-e4b)

## Quick Start
```bash
pip install unsloth gradio
python app.py
```

## Training Details (Unsloth QLoRA)
| Metric | Value |
|--------|-------|
| Base model | unsloth/gemma-4-E4B-it |
| Method | QLoRA (4-bit, LoRA r=16) |
| Training examples | 638 |
| Final loss | 0.6003 |
| Peak VRAM | 13.92 GB |
| Training time | 74.5 minutes |
| Platform | Google Colab T4 GPU |
| Framework | Unsloth + Transformers |

## Dataset Categories
| Category | Examples |
|----------|----------|
| Jan Aushadhi savings | 348 |
| NSQ/Spurious alerts | 115 |
| Pharmacist fraud detection | 108 |
| Burns/Fire first aid | 56 |
| Emergency first aid | 26 |
| Safety refusals | 12 |
| Hindi Devanagari | 12 |

## Why MedSaathi
India has 1.4 billion people. Most cannot verify if a pharmacist's 
substitute is safe. Most cannot read medical jargon. Most cannot 
afford branded medicines when generics exist at 70-90% lower cost.

MedSaathi solves all three — in their language, offline if needed.

## Safety Architecture
- RAG-first: verified database before LLM
- Training-time filtering: zero dose advice in training data
- Post-generation safety check: dose patterns intercepted

## License
MIT
