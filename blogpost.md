# Introduction
```An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well.```

Transformer-based autoregressive language models have been shown to achieve remarkable performance on open-ended language generation tasks. However, the large scale of these models limits their applicability in real-world scenarios. Early-exit approaches attempt to address this constraint by allowing the model to skip computations at later layers whenever the confidence of the current prediction is sufficient. Although methods like BERxiT have existed for encoder-based models, Confident Adaptive Language Modeling (CALM) is the first work focusing on autoregressive LLMs. Without significant loss in performance, this framework has been demonstrated to allow for up to x3 speedup in inference time. Additional improvements can be obtained using simple linear mapping to cast intermediate representations as final-layer representations.

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

