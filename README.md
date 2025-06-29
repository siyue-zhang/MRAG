# MRAG: A Modular Retrieval Framework for Time-Sensitive Question Answering

[**📖 arXiv**](https://arxiv.org/abs/2412.15540) | [**🤗 Dataset**](https://huggingface.co/datasets/siyue/TempRAGEval)


**Temp**oral QA for **RAG** **Eval**uation benchmark (TempRAGEval) is a evaluation benchmark of time-sensitive questions. We repurpose two existing datasets:
- **TimeQA** (Chen et al., 2021)
- **SituatedQA** (Zhang and Choi, 2021)

We manually augment the test questions with temporal perturbations (e.g., modifying the time period). For example, we change `2019` to `6 May 2021` while preserving the answer `Boris Johnson`.

![](./img/idea.png)


In addition, we annotate gold evidence on Wikipedia for more accurate retrieval evaluation. We provide max two gold sentences. If the Wikipedia text chunk contains any one of gold sentences, the text chunk is regarded as gold evidence. It is guaranteed that gold sentences are from the Wikipedia corpus. `corpora/wiki/enwiki-dec2021` is the only corpus used in the benchmark, consisting of 33.1M text chunks from [ATLAS](https://github.com/facebookresearch/atlas).

The objective is to evaluate temporal reasoning capabilities and robustness for both retrieval systems and LLMs.

Notes: the samples with empty `time_relation` are excluded in the evaluation.

You can download this dataset by the following command:
```python
from datasets import load_dataset

dataset = load_dataset("siyue/TempRAGEval")

# print the first example on the test set
print(dataset["test"][0])
```

## Contact

For any issues or questions, kindly email us at: Siyue Zhang (siyue001@e.ntu.edu.sg).

## Citation

If you use the dataset in your work, please kindly cite the paper:
```
@misc{siyue2024mragmodularretrievalframework,
      title={MRAG: A Modular Retrieval Framework for Time-Sensitive Question Answering}, 
      author={Zhang Siyue and Xue Yuxiang and Zhang Yiming and Wu Xiaobao and Luu Anh Tuan and Zhao Chen},
      year={2024},
      eprint={2412.15540},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.15540}, 
}
```