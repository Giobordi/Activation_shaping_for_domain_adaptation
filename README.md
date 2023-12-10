## DOMAIN ADAPTATION VIA ACTIVATION SHAPING

One of the biggest limitations of deep models is their inability to work consistently well with
any data met at test time. The training and the test set could be characterized by different
data distributions (illumination conditions, visual style, background, etc.) leading to an
inevitable decrease in the model performances. This is known as the problem of domain
shift and it’s one of the most investigated topics in the computer vision community. \  It can be
formalized with the setting of Unsupervised Domain Adaptation (UDA) where there is a
labeled Source domain (Training Set) and an unlabeled Target domain (Test Set), the goal is
to train a model that works well on the target distribution

![immagine](https://github.com/Giobordi/Activation_shaping_for_domain_adaptation/assets/129875197/5f9e7749-353b-4efc-b84e-325c8c85954a)

Activation Shaping is the process of modifying activation maps from a layer within an
architecture using either some user-defined or learned rules and is highly effective for detecting out-of-distribution (OOD) data.In this
project, we aim to leverage the benefits of this OOD detection strategy to exploit the insights
regarding data distribution discrepancies with the goal of learning a model that is robust to
domain shift.
