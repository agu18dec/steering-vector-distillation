# Subliminal Learning via Paraphrasing

![Overview Figure](Overview.png)

This github repo is part of the ARENA 6.0 hackathon (we won the hackathon with this!), and our ARENA capstone project. 
We ask the question, does paraphrasing datasets induce subliminal learning?
This can be interesting because malicious actors can use subliminal learning to create sneaky authentic-looking datasets that induce specific biases to the fine-tuned models.
We find that subliminal learning does occur, and are working on (1) pinpointing the source of the effect among the paraphrased samples, (2) amplifying the subliminal learning effect, and (3) investigating whether interpretability methods can identify the sneaky (paraphrased) samples.

- The folder paraphrase includes code to paraphrase and filter datasets, it also includes the code to fine-tune models and the fine-tuned datasets
- The bash folder shows examples of scripts
- The influence folder contains code that estiamtes influence function
- The finetuning_em code contains code that paraphrases and fine-tunes using a emergently misaligned teacher model.
- You can generate the em paraphrasings with paraphrase_em.py, filter the dataset filter_gpt_em.py, then sft with sft_train_em.py. 

Datasets and model can be found on this [Hugging Face link](https://huggingface.co/collections/Taywon/subliminal-learning-paraphrase-68da5f0f3dceab47c30817dd) and here https://huggingface.co/collections/matboz/subliminal-learning-paraphrasing-2




