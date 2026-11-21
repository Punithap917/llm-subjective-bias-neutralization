Detection and Neutralization of Subjective Bias in Text

A two-stage pipeline that finds biased phrases in a sentence and rewrites the sentence in a neutral tone with a large language model (LLM), while keeping its facts and meaning. We compare several prompting strategies across open and commercial LLMs on the Wikipedia Neutrality Corpus (WNC).

Group project (6 members) for AI6130 Large Language Models, MSc Artificial Intelligence, NTU Singapore (2026).

📄 Read the final report · ▶️ Watch the presentation video

The problem

Biased wording like "controversial federal marriage amendment" subtly pushes a reader's opinion. The goal is to rewrite it neutrally ("federal marriage amendment") without changing the facts. Simple paraphrasing isn't enough. The edit needs to be precise: remove the loaded words and leave everything else alone.

Approach

Stage A: Bias tagger. A BiLSTM model tags biased spans in a sentence (BIO tagging).

Stage B: LLM rewriter. An LLM rewrites the sentence, guided by one of these prompting strategies:

Strategy	Idea
Zero-shot	Just ask the model to neutralize the sentence
Few-shot	Show 3 example rewrites first
With-bias-tags	Mark the biased words found by the tagger with <bias> tags
NPOV	Ask the model to follow Wikipedia's neutral point of view policy
Self-refine	Generate, critique, then improve
Context Enriched (CE), our method	Multi-turn prompt combining tagged few-shot examples with <bias> markers

Models tested: LLaMA-3-8B, Qwen-2.5-7B, Qwen3.5 (9B / 27B / Flash), Claude Haiku / Sonnet / Opus, DeepSeek-V3.2

Metrics: BLEU, METEOR, BERTScore, bias phrase retention, over-edit rate, and a combined Neutralization F-score (NF1)

Key findings
Two missing pieces of information explain the trade-off. Few-shot keeps meaning well but removes little bias. Bias tags remove bias but over-edit. The model needs to know both where to edit (localization) and how much to edit (calibration).
Our Context Enriched prompt supplies both and scores best: NF1 = 0.775 on DeepSeek-V3.2. It beats with-bias-tags on both axes, and beats few-shot on debiasing by 16 percentage points.
Stricter rule wording reduces over-editing, confirmed with a paired-bootstrap test.
Reasoning matters more than model size: chain-of-thought reduces failed debiasing by about 10 pp, while going from 9B to 27B parameters helps only 1.3 pp.
Prompt gains don't automatically transfer: CE performs worse on Claude Sonnet via a single-turn CLI, so prompt engineering results depend on the setup.

Results on DeepSeek-V3.2 (n = 500):

Strategy	Failed debiasing ↓	BERTScore ↑	Over-edit ↓	NF1 ↑
Zero-shot	46.0%	0.659	0.194	0.607
With-bias-tags	31.4%	0.611	0.234	0.646
Few-shot	44.2%	0.832	0.094	0.679
CE (constrained)	25.8%	0.811	0.089	0.775
Team

Raygenius Luckschild · Pramod Sanjay · Shong Yee Kiat · Veluchamy Punitha · Sun Junkai · Zhao Huiting

Repository structure
├── llm_prompting_runner/
│   ├── run_local_models.py          # Run prompting strategies on open models (OpenRouter)
│   ├── run_claude.py                # Run strategies on Claude models
│   ├── rescore_outputs_inplace.py   # Compute all evaluation metrics
│   ├── tagging/                     # BiLSTM bias tagger (model, data, training)
│   ├── outputs_llama/ outputs_qwen/ outputs_claude/   # Model outputs + summaries
│   └── README.md                    # Detailed setup and run commands
├── report/
│   └── AI6130_Final_Report.pdf
└── README.md
How to run
bash
pip install -r llm_prompting_runner/requirements.txt
python llm_prompting_runner/run_local_models.py --api_base https://openrouter.ai/api/v1 --models qwen/qwen-2.5-7b-instruct --samples 200

You need an OpenRouter API key. See llm_prompting_runner/README.md for all options.

Tech stack

Python · PyTorch · Hugging Face Transformers · OpenAI SDK / OpenRouter · BERTScore · sacreBLEU · NLTK

Forked from the team repository comicsans-02/AI6130_LLM_Project.
