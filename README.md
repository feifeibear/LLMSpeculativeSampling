# Fast inference from transformers via speculative decoding

This repo implement a speculative sampling for large lauguage model(LLM) decoding.
It uses two models during decoding: a target model and an approximation model.
The apporximation model is a smaller model and target model is a larger one.
The apporximation model guess tokens and the target model corrects the guesses.
Therefore, it decodes by running the target model in parallel on the outputs of the approximation models, which is more efficient than decoding with the target model alone.

## Usage
In the sample, I use [bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1/tree/main) as the target model, [bloom-560m](https://huggingface.co/bigscience/bloom-560m/tree/main) as the approximation model.

```bash
python sample.py --input "The quick brown fox jumps over the lazy " --target_model bigscience/bloomz-7b1 --approximation_model bigscience/bloom-560m
```


## References
```
@inproceedings{leviathan2023fast,
  title={Fast inference from transformers via speculative decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  booktitle={International Conference on Machine Learning},
  pages={19274--19286},
  year={2023},
  organization={PMLR}
}
```