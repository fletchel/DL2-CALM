# Introduction
```An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well.```

Transformer-based autoregressive language models have been shown to achieve remarkable performance on open-ended language generation tasks. However, the large scale of these models limits their applicability in real-world scenarios. Early-exit approaches attempt to address this constraint by allowing the model to skip computations at later layers whenever the confidence of the current prediction is sufficient. Although methods like BERxiT have existed for encoder-based models, Confident Adaptive Language Modeling (CALM) is the first work focusing on autoregressive LLMs. Without significant loss in performance, this framework has been demonstrated to allow for up to x3 speedup in inference time. Additional improvements can be obtained using simple linear mapping to cast intermediate representations as final-layer representations.

## Related work

Improving the efficiency of LLMs is being studied in several different ways. Model quantization approaches reduce model size by decreasing the precision of the weights, and also speed up inference by reducing the precision to which the activations are calculated (https://arxiv.org/abs/1511.06393).  Another method is pruning, in which unimportant parts of the model may be removed (https://arxiv.org/pdf/1506.02626). Knowledge distillation uses outputs from the large 'teacher' model (the original trained model) to train a smaller 'student' model to produce similar outputs with fewer parameters (https://arxiv.org/abs/1503.02531). 

The approach studied in the CALM paper is *early exiting*, where the model can decide to stop processing a token without having passed the token through every layer of the transformer. Early exiting was first proposed for transformers used for classification by Schwartz et al. (2020) (https://arxiv.org/pdf/2004.07453). Then, Xin et al. (2021) (https://aclanthology.org/2021.eacl-main.8.pdf) introduced BERxiT to deal with more general tasks than just classification. Recently, it has been shown that the top-ranked prediction for a token often does not change after a certain layer (https://arxiv.org/abs/2203.14680), motivating both the CALM paper and our choice for an extension in which we decide to only propagate the top-ranked tokens to subsequent layers.


# Review
```Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response```

Further gains in the efficiency of autoregressive LLMs would increase the accessibility of these models for applications in both academia and industry. Although the CALM framework already provides a noticeable improvement, the best performing confidence estimation method is inefficient with a computational complexity of $O(vd)$. Additionally, the impact of analysing the full history of representations generated at previous layers rather than only the current layer has not been analysed. Research focused on these two points could lead to further reductions in computational costs connected when running autoregressive models.

# Contribution
```Describe your novel contribution.```

We have implemented a new method for early-exiting: an attention layer. We have also tried limiting the vocabulary of the softmax classifier to high confidence tokens in early layers.

# Results
```Results of your work (link that part with the code in the jupyter notebook)```

We are just in the process of getting results, so sadly we do not have results yet.

# Conclusion
```Conclude```


# Contributions per student
```Close the notebook with a description of each student's contribution.```

